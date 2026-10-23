//! Exact existing PRSCODE1 serialization into a bounded caller-owned sink.
//! Every source collection in admitted native owners declares its length.
//! An unknown-length custom serializer is refused, never buffered without bound.
use super::*;
use std::io::Write;
use serde::ser::Error as _;

pub fn write_to<T:Serialize+?Sized>(value:&T,writer:&mut dyn Write,limit:u64)->Result<u64> {
    let expected=encoded_size(value,limit)?;
    let mut encoder=StreamEncoder{writer,count:0,limit,depth:0};encoder.write(MAGIC)?;
    value.serialize(&mut encoder)?;
    if encoder.count!=expected{return Err(err("source changed after exact stream sizing"));}Ok(expected)
}
struct StreamEncoder<'a>{writer:&'a mut dyn Write,count:u64,limit:u64,depth:usize}
impl<'b> StreamEncoder<'b> {
    fn write(&mut self,bytes:&[u8])->Result<()> {
        let next=self.count.checked_add(bytes.len() as u64).ok_or_else(||err("typed stream overflow"))?;
        if next>self.limit{return Err(err("typed stream exceeds admission"));}
        self.writer.write_all(bytes).map_err(Error::custom)?;self.count=next;Ok(())
    }
    fn word(&mut self,tag:u8,bytes:&[u8])->Result<()> {self.write(&[tag])?;self.write(bytes)}
    fn string(&mut self,value:&str)->Result<()> {self.word(10,&(value.len() as u64).to_le_bytes())?;self.write(value.as_bytes())}
    fn variant(&mut self,value:&str)->Result<()> {self.write(&[16])?;self.string(value)}
    fn compound<'a>(&'a mut self,tag:u8,length:Option<usize>)->Result<StreamCompound<'a,'b>> {
        let expected=length.ok_or_else(||err("native streaming field has an uncounted custom collection"))?;
        if self.depth>=MAX_DEPTH{return Err(err("host-state nesting limit"));}
        self.word(tag,&(expected as u64).to_le_bytes())?;self.depth+=1;
        Ok(StreamCompound{s:self,expected,count:0,key_pending:false})
    }
}
struct StreamCompound<'a,'b>{s:&'a mut StreamEncoder<'b>,expected:usize,count:usize,key_pending:bool}
impl StreamCompound<'_,'_> {
    fn element<T:Serialize+?Sized>(&mut self,value:&T)->Result<()> {value.serialize(&mut*self.s)?;self.count+=1;Ok(())}
    fn field<T:Serialize+?Sized>(&mut self,name:&str,value:&T)->Result<()> {self.s.string(name)?;self.element(value)}
    fn finish(self)->Result<()> {if self.key_pending||self.count!=self.expected{return Err(err("typed stream collection count changed"));}self.s.depth-=1;Ok(())}
}
macro_rules! signed_stream {($($name:ident:$kind:ty),*)=>{$(fn $name(self,value:$kind)->Result<()> {self.word(3,&(value as i64).to_le_bytes())})*}}
macro_rules! unsigned_stream {($($name:ident:$kind:ty),*)=>{$(fn $name(self,value:$kind)->Result<()> {self.word(4,&(value as u64).to_le_bytes())})*}}
impl<'a,'b> serde::Serializer for &'a mut StreamEncoder<'b> {
    type Ok=();type Error=Error;type SerializeSeq=StreamCompound<'a,'b>;type SerializeTuple=StreamCompound<'a,'b>;
    type SerializeTupleStruct=StreamCompound<'a,'b>;type SerializeTupleVariant=StreamCompound<'a,'b>;type SerializeMap=StreamCompound<'a,'b>;
    type SerializeStruct=StreamCompound<'a,'b>;type SerializeStructVariant=StreamCompound<'a,'b>;
    fn serialize_bool(self,value:bool)->Result<()> {self.write(&[if value{2}else{1}])}
    signed_stream!(serialize_i8:i8,serialize_i16:i16,serialize_i32:i32,serialize_i64:i64);
    unsigned_stream!(serialize_u8:u8,serialize_u16:u16,serialize_u32:u32,serialize_u64:u64);
    fn serialize_i128(self,value:i128)->Result<()>{self.word(5,&value.to_le_bytes())}
    fn serialize_u128(self,value:u128)->Result<()>{self.word(6,&value.to_le_bytes())}
    fn serialize_f32(self,value:f32)->Result<()>{self.word(7,&value.to_bits().to_le_bytes())}
    fn serialize_f64(self,value:f64)->Result<()>{self.word(8,&value.to_bits().to_le_bytes())}
    fn serialize_char(self,value:char)->Result<()>{self.word(9,&(value as u32).to_le_bytes())}
    fn serialize_str(self,value:&str)->Result<()>{self.string(value)}
    fn serialize_bytes(self,value:&[u8])->Result<()>{self.word(11,&(value.len() as u64).to_le_bytes())?;self.write(value)}
    fn serialize_none(self)->Result<()>{self.write(&[12])}
    fn serialize_some<T:Serialize+?Sized>(self,value:&T)->Result<()>{self.write(&[13])?;value.serialize(self)}
    fn serialize_unit(self)->Result<()>{self.write(&[0])}
    fn serialize_unit_struct(self,_:&'static str)->Result<()>{self.serialize_unit()}
    fn serialize_unit_variant(self,_:&'static str,_:u32,variant:&'static str)->Result<()>{self.variant(variant)?;self.serialize_unit()}
    fn serialize_newtype_struct<T:Serialize+?Sized>(self,_:&'static str,value:&T)->Result<()>{value.serialize(self)}
    fn serialize_newtype_variant<T:Serialize+?Sized>(self,_:&'static str,_:u32,variant:&'static str,value:&T)->Result<()>{self.variant(variant)?;value.serialize(self)}
    fn serialize_seq(self,length:Option<usize>)->Result<Self::SerializeSeq>{self.compound(14,length)}
    fn serialize_tuple(self,length:usize)->Result<Self::SerializeTuple>{self.compound(14,Some(length))}
    fn serialize_tuple_struct(self,_:&'static str,length:usize)->Result<Self::SerializeTupleStruct>{self.compound(14,Some(length))}
    fn serialize_tuple_variant(self,_:&'static str,_:u32,variant:&'static str,length:usize)->Result<Self::SerializeTupleVariant>{self.variant(variant)?;self.compound(14,Some(length))}
    fn serialize_map(self,length:Option<usize>)->Result<Self::SerializeMap>{self.compound(15,length)}
    fn serialize_struct(self,_:&'static str,length:usize)->Result<Self::SerializeStruct>{self.compound(15,Some(length))}
    fn serialize_struct_variant(self,_:&'static str,_:u32,variant:&'static str,length:usize)->Result<Self::SerializeStructVariant>{self.variant(variant)?;self.compound(15,Some(length))}
    fn is_human_readable(&self)->bool{false}
}
impl SerializeSeq for StreamCompound<'_,'_>{type Ok=();type Error=Error;fn serialize_element<T:Serialize+?Sized>(&mut self,v:&T)->Result<()>{self.element(v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeTuple for StreamCompound<'_,'_>{type Ok=();type Error=Error;fn serialize_element<T:Serialize+?Sized>(&mut self,v:&T)->Result<()>{self.element(v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeTupleStruct for StreamCompound<'_,'_>{type Ok=();type Error=Error;fn serialize_field<T:Serialize+?Sized>(&mut self,v:&T)->Result<()>{self.element(v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeTupleVariant for StreamCompound<'_,'_>{type Ok=();type Error=Error;fn serialize_field<T:Serialize+?Sized>(&mut self,v:&T)->Result<()>{self.element(v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeStruct for StreamCompound<'_,'_>{type Ok=();type Error=Error;fn serialize_field<T:Serialize+?Sized>(&mut self,k:&'static str,v:&T)->Result<()>{self.field(k,v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeStructVariant for StreamCompound<'_,'_>{type Ok=();type Error=Error;fn serialize_field<T:Serialize+?Sized>(&mut self,k:&'static str,v:&T)->Result<()>{self.field(k,v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeMap for StreamCompound<'_,'_>{type Ok=();type Error=Error;
    fn serialize_key<T:Serialize+?Sized>(&mut self,key:&T)->Result<()>{if self.key_pending{return Err(err("map key without value"));}key.serialize(&mut*self.s)?;self.key_pending=true;Ok(())}
    fn serialize_value<T:Serialize+?Sized>(&mut self,value:&T)->Result<()>{if !self.key_pending{return Err(err("map value without key"));}self.element(value)?;self.key_pending=false;Ok(())}
    fn end(self)->Result<()>{self.finish()}}
