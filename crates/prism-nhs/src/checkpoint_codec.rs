//! Versioned, self-describing Serde wire format for scientific host state.
//! Float words are written as their original IEEE bits, including signed zero,
//! infinities and NaN payloads. Unlike JSON it never replaces nonfinite values;
//! unlike untyped bincode it supports serde_json::Value and tagged enums through
//! deserialize_any. It records no Rust object memory, pointers or padding.
use serde::{Serialize, Deserialize};
use serde::ser::{SerializeSeq, SerializeTuple, SerializeTupleStruct, SerializeTupleVariant, SerializeMap, SerializeStruct, SerializeStructVariant};
use serde::de::{DeserializeSeed, Visitor, SeqAccess, MapAccess, EnumAccess, VariantAccess, IntoDeserializer};
use std::fmt;

mod bounded;
mod streaming;
mod reader;
pub use reader::{from_reader,from_reader_with_allocation_budget};
pub use streaming::write_to;
pub use bounded::{allocation_plan,encoded_size,to_bytes_bounded};

const MAGIC: &[u8;8] = b"PRSCODE1";
const MAX_DEPTH:usize=256;
#[derive(Debug)]
pub struct Error(String);
impl fmt::Display for Error { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {f.write_str(&self.0)} }
impl std::error::Error for Error {}
impl serde::ser::Error for Error {fn custom<T:fmt::Display>(m:T)->Self {Self(m.to_string())}}
impl serde::de::Error for Error {fn custom<T:fmt::Display>(m:T)->Self {Self(m.to_string())}}
type Result<T> = std::result::Result<T,Error>;
fn err(m:&str)->Error {Error(m.into())}

pub fn to_bytes<T:Serialize+?Sized>(value:&T)->Result<Vec<u8>> {
    let mut s=Encoder{bytes:MAGIC.to_vec(),depth:0};value.serialize(&mut s)?;Ok(s.bytes)
}
pub fn from_bytes<'de,T:Deserialize<'de>>(bytes:&'de [u8])->Result<T> {
    if bytes.get(..8)!=Some(MAGIC.as_slice()) {return Err(err("invalid scientific host-state codec"));}
    let mut d=Decoder{bytes,at:8,depth:0};let value=T::deserialize(&mut d)?;
    if d.at!=bytes.len(){return Err(err("trailing scientific host-state bytes"));}Ok(value)
}
struct Encoder {bytes:Vec<u8>,depth:usize}
impl Encoder {
    fn word(&mut self,tag:u8,bytes:&[u8]) {self.bytes.push(tag);self.bytes.extend_from_slice(bytes)}
    fn string(&mut self,v:&str) {self.word(10,&(v.len()as u64).to_le_bytes());self.bytes.extend_from_slice(v.as_bytes())}
    fn compound(&mut self,tag:u8,expected:Option<usize>)->Result<Compound<'_>> {
        if self.depth>=MAX_DEPTH{return Err(err("host-state nesting limit"));}
        self.depth+=1;self.bytes.push(tag);let length_offset=self.bytes.len();self.bytes.extend_from_slice(&[0;8]);
        Ok(Compound{s:self,length_offset,count:0,expected,key_pending:false})
    }
    fn variant(&mut self,variant:&str){self.bytes.push(16);self.string(variant)}
}
struct Compound<'a>{s:&'a mut Encoder,length_offset:usize,count:usize,expected:Option<usize>,key_pending:bool}
impl Compound<'_> {
    fn finish(self)->Result<()> {
        if self.key_pending||self.expected.is_some_and(|x|x!=self.count){return Err(err("serialized collection length mismatch"));}
        self.s.bytes[self.length_offset..self.length_offset+8].copy_from_slice(&(self.count as u64).to_le_bytes());
        self.s.depth-=1;Ok(())
    }
    fn element<T:Serialize+?Sized>(&mut self,v:&T)->Result<()> {v.serialize(&mut*self.s)?;self.count+=1;Ok(())}
    fn field<T:Serialize+?Sized>(&mut self,k:&str,v:&T)->Result<()> {self.s.string(k);self.element(v)}
}
macro_rules! signed {($($name:ident:$t:ty),*)=>{$(fn $name(self,v:$t)->Result<()> {self.word(3,&(v as i64).to_le_bytes());Ok(())})*}}
macro_rules! unsigned {($($name:ident:$t:ty),*)=>{$(fn $name(self,v:$t)->Result<()> {self.word(4,&(v as u64).to_le_bytes());Ok(())})*}}
impl<'a> serde::Serializer for &'a mut Encoder {
    type Ok=();type Error=Error;type SerializeSeq=Compound<'a>;type SerializeTuple=Compound<'a>;
    type SerializeTupleStruct=Compound<'a>;type SerializeTupleVariant=Compound<'a>;type SerializeMap=Compound<'a>;
    type SerializeStruct=Compound<'a>;type SerializeStructVariant=Compound<'a>;
    fn serialize_bool(self,v:bool)->Result<()> {self.bytes.push(if v{2}else{1});Ok(())}
    signed!(serialize_i8:i8,serialize_i16:i16,serialize_i32:i32,serialize_i64:i64);
    unsigned!(serialize_u8:u8,serialize_u16:u16,serialize_u32:u32,serialize_u64:u64);
    fn serialize_i128(self,v:i128)->Result<()> {self.word(5,&v.to_le_bytes());Ok(())}
    fn serialize_u128(self,v:u128)->Result<()> {self.word(6,&v.to_le_bytes());Ok(())}
    fn serialize_f32(self,v:f32)->Result<()> {self.word(7,&v.to_bits().to_le_bytes());Ok(())}
    fn serialize_f64(self,v:f64)->Result<()> {self.word(8,&v.to_bits().to_le_bytes());Ok(())}
    fn serialize_char(self,v:char)->Result<()> {self.word(9,&(v as u32).to_le_bytes());Ok(())}
    fn serialize_str(self,v:&str)->Result<()> {self.string(v);Ok(())}
    fn serialize_bytes(self,v:&[u8])->Result<()> {self.word(11,&(v.len()as u64).to_le_bytes());self.bytes.extend_from_slice(v);Ok(())}
    fn serialize_none(self)->Result<()> {self.bytes.push(12);Ok(())}
    fn serialize_some<T:Serialize+?Sized>(self,v:&T)->Result<()> {self.bytes.push(13);v.serialize(self)}
    fn serialize_unit(self)->Result<()> {self.bytes.push(0);Ok(())}
    fn serialize_unit_struct(self,_:&'static str)->Result<()> {self.serialize_unit()}
    fn serialize_unit_variant(self,_:&'static str,_:u32,v:&'static str)->Result<()> {self.variant(v);self.serialize_unit()}
    fn serialize_newtype_struct<T:Serialize+?Sized>(self,_:&'static str,v:&T)->Result<()> {v.serialize(self)}
    fn serialize_newtype_variant<T:Serialize+?Sized>(self,_:&'static str,_:u32,n:&'static str,v:&T)->Result<()> {self.variant(n);v.serialize(self)}
    fn serialize_seq(self,n:Option<usize>)->Result<Self::SerializeSeq> {self.compound(14,n)}
    fn serialize_tuple(self,n:usize)->Result<Self::SerializeTuple> {self.compound(14,Some(n))}
    fn serialize_tuple_struct(self,_:&'static str,n:usize)->Result<Self::SerializeTupleStruct> {self.compound(14,Some(n))}
    fn serialize_tuple_variant(self,_:&'static str,_:u32,v:&'static str,n:usize)->Result<Self::SerializeTupleVariant> {self.variant(v);self.compound(14,Some(n))}
    fn serialize_map(self,n:Option<usize>)->Result<Self::SerializeMap> {self.compound(15,n)}
    fn serialize_struct(self,_:&'static str,n:usize)->Result<Self::SerializeStruct> {self.compound(15,Some(n))}
    fn serialize_struct_variant(self,_:&'static str,_:u32,v:&'static str,n:usize)->Result<Self::SerializeStructVariant> {self.variant(v);self.compound(15,Some(n))}
    fn is_human_readable(&self)->bool {false}
}
impl SerializeSeq for Compound<'_>{type Ok=();type Error=Error;fn serialize_element<T:Serialize+?Sized>(&mut self,v:&T)->Result<()>{self.element(v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeTuple for Compound<'_>{type Ok=();type Error=Error;fn serialize_element<T:Serialize+?Sized>(&mut self,v:&T)->Result<()>{self.element(v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeTupleStruct for Compound<'_>{type Ok=();type Error=Error;fn serialize_field<T:Serialize+?Sized>(&mut self,v:&T)->Result<()>{self.element(v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeTupleVariant for Compound<'_>{type Ok=();type Error=Error;fn serialize_field<T:Serialize+?Sized>(&mut self,v:&T)->Result<()>{self.element(v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeStruct for Compound<'_>{type Ok=();type Error=Error;fn serialize_field<T:Serialize+?Sized>(&mut self,k:&'static str,v:&T)->Result<()>{self.field(k,v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeStructVariant for Compound<'_>{type Ok=();type Error=Error;fn serialize_field<T:Serialize+?Sized>(&mut self,k:&'static str,v:&T)->Result<()>{self.field(k,v)}fn end(self)->Result<()>{self.finish()}}
impl SerializeMap for Compound<'_>{
    type Ok=();type Error=Error;
    fn serialize_key<T:Serialize+?Sized>(&mut self,k:&T)->Result<()> {if self.key_pending{return Err(err("map key without value"));}k.serialize(&mut*self.s)?;self.key_pending=true;Ok(())}
    fn serialize_value<T:Serialize+?Sized>(&mut self,v:&T)->Result<()> {if !self.key_pending{return Err(err("map value without key"));}self.element(v)?;self.key_pending=false;Ok(())}
    fn end(self)->Result<()> {self.finish()}
}
struct Decoder<'de>{bytes:&'de[u8],at:usize,depth:usize}
impl<'de> Decoder<'de>{
    fn take(&mut self,n:usize)->Result<&'de[u8]>{let end=self.at.checked_add(n).ok_or_else(||err("length overflow"))?;
        let out=self.bytes.get(self.at..end).ok_or_else(||err("truncated scientific state"))?;self.at=end;Ok(out)}
    fn tag(&mut self)->Result<u8>{Ok(self.take(1)?[0])}
    fn length(&mut self)->Result<usize>{let n=u64::from_le_bytes(self.take(8)?.try_into().unwrap());let n=usize::try_from(n).map_err(|_|err("host-state size overflow"))?;
        if n>self.bytes.len()-self.at{return Err(err("host-state length exceeds remaining input"));}Ok(n)}
    fn string(&mut self)->Result<&'de str>{if self.tag()?!=10{return Err(err("expected scientific state string"));}let n=self.length()?;std::str::from_utf8(self.take(n)?).map_err(|_|err("invalid state UTF-8"))}
}
impl<'de> serde::Deserializer<'de> for &mut Decoder<'de>{
    type Error=Error;
    fn deserialize_any<V:Visitor<'de>>(self,v:V)->Result<V::Value>{
        if self.depth>=MAX_DEPTH{return Err(err("host-state nesting limit"));}self.depth+=1;
        let out=match self.tag()? {
            0=>v.visit_unit(),1=>v.visit_bool(false),2=>v.visit_bool(true),
            3=>v.visit_i64(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            4=>v.visit_u64(u64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            5=>v.visit_i128(i128::from_le_bytes(self.take(16)?.try_into().unwrap())),
            6=>v.visit_u128(u128::from_le_bytes(self.take(16)?.try_into().unwrap())),
            7=>v.visit_f32(f32::from_bits(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))),
            8=>v.visit_f64(f64::from_bits(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))),
            9=>v.visit_char(char::from_u32(u32::from_le_bytes(self.take(4)?.try_into().unwrap())).ok_or_else(||err("invalid character"))?),
            10=>{let n=self.length()?;v.visit_borrowed_str(std::str::from_utf8(self.take(n)?).map_err(|_|err("invalid state UTF-8"))?)},
            11=>{let n=self.length()?;v.visit_borrowed_bytes(self.take(n)?)},
            12=>v.visit_none(),13=>v.visit_some(&mut*self),
            14=>{let n=self.length()?;let mut seq=Sequence{d:self,remaining:n};let out=v.visit_seq(&mut seq)?;
                if seq.remaining!=0{return Err(err("unconsumed sequence state"));}Ok(out)},
            15=>{let n=self.length()?;let mut map=Mapping{d:self,remaining:n,pending:false};let out=v.visit_map(&mut map)?;
                if map.remaining!=0||map.pending{return Err(err("unconsumed map state"));}Ok(out)},
            16=>{let variant=self.string()?;v.visit_enum(Enumeration{d:self,variant})},
            _=>Err(err("unknown scientific state tag")),
        };self.depth-=1;out
    }
    fn deserialize_option<V:Visitor<'de>>(self,v:V)->Result<V::Value>{match self.tag()?{12=>v.visit_none(),13=>v.visit_some(self),_=>Err(err("expected optional state tag"))}}
    fn deserialize_newtype_struct<V:Visitor<'de>>(self,_:&'static str,v:V)->Result<V::Value>{v.visit_newtype_struct(self)}
    fn deserialize_enum<V:Visitor<'de>>(self,_:&'static str,_:&'static[&'static str],v:V)->Result<V::Value>{
        if self.tag()?!=16{return Err(err("expected enum state tag"));}let variant=self.string()?;v.visit_enum(Enumeration{d:self,variant})}
    fn is_human_readable(&self)->bool{false}
    serde::forward_to_deserialize_any!{bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf unit unit_struct seq tuple tuple_struct map struct identifier ignored_any}
}
struct Sequence<'a,'de>{d:&'a mut Decoder<'de>,remaining:usize}
impl<'de> SeqAccess<'de> for &mut Sequence<'_,'de>{type Error=Error;
    fn next_element_seed<T:DeserializeSeed<'de>>(&mut self,seed:T)->Result<Option<T::Value>>{
        if self.remaining==0{return Ok(None);}self.remaining-=1;Ok(Some(seed.deserialize(&mut*self.d)?))}
    fn size_hint(&self)->Option<usize>{Some(self.remaining)}
}
struct Mapping<'a,'de>{d:&'a mut Decoder<'de>,remaining:usize,pending:bool}
impl<'de> MapAccess<'de> for &mut Mapping<'_,'de>{type Error=Error;
    fn next_key_seed<K:DeserializeSeed<'de>>(&mut self,seed:K)->Result<Option<K::Value>>{
        if self.pending{return Err(err("map value not consumed"));}if self.remaining==0{return Ok(None);}
        self.pending=true;Ok(Some(seed.deserialize(&mut*self.d)?))}
    fn next_value_seed<V:DeserializeSeed<'de>>(&mut self,seed:V)->Result<V::Value>{
        if !self.pending{return Err(err("map value lacks key"));}let value=seed.deserialize(&mut*self.d)?;self.pending=false;self.remaining-=1;Ok(value)}
    fn size_hint(&self)->Option<usize>{Some(self.remaining)}
}
struct Enumeration<'a,'de>{d:&'a mut Decoder<'de>,variant:&'de str}
impl<'a,'de> EnumAccess<'de> for Enumeration<'a,'de>{type Error=Error;type Variant=Self;
    fn variant_seed<V:DeserializeSeed<'de>>(self,seed:V)->Result<(V::Value,Self)>{Ok((seed.deserialize(self.variant.into_deserializer())?,self))}
}
impl<'de> VariantAccess<'de> for Enumeration<'_,'de>{type Error=Error;
    fn unit_variant(self)->Result<()>{Deserialize::deserialize(self.d)}
    fn newtype_variant_seed<T:DeserializeSeed<'de>>(self,seed:T)->Result<T::Value>{seed.deserialize(self.d)}
    fn tuple_variant<V:Visitor<'de>>(self,_:usize,v:V)->Result<V::Value>{serde::Deserializer::deserialize_seq(self.d,v)}
    fn struct_variant<V:Visitor<'de>>(self,_:&'static[&'static str],v:V)->Result<V::Value>{serde::Deserializer::deserialize_map(self.d,v)}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug,Serialize,Deserialize)]
    struct ScientificState { a:Vec<f32>,b:Vec<f64>,count:u128,condition:serde_json::Value,phase:Phase,missing:Option<u64> }
    #[derive(Debug,Serialize,Deserialize,PartialEq)] enum Phase{Cold,Sample(u64),Coupled{parity:u8,seed:u64},History(u32,f64)}
    #[test]fn float_bits_and_json_value_are_lossless(){
        let a=[0,0x80000000,0x7f800000,0xff800000,0x7fcabcde,0x7f812345,1,0x3f800001].map(f32::from_bits).to_vec();
        let b=[0,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8abcd12345678,1].map(f64::from_bits).to_vec();
        let s=ScientificState{a,b,count:u128::MAX,condition:serde_json::json!({"nested":[1,2.25,null,true]}),phase:Phase::Coupled{parity:1,seed:42},missing:None};
        let bytes=to_bytes(&s).unwrap();let out:ScientificState=from_bytes(&bytes).unwrap();
        assert_eq!(s.a.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),out.a.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        assert_eq!(s.b.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),out.b.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        assert_eq!(s.condition,out.condition);assert_eq!(s.phase,out.phase);assert_eq!(s.count,out.count);
        assert_eq!(to_bytes(&out).unwrap(),bytes);
    }
    #[test]fn all_enum_shapes_and_truncation_refusal(){
        for value in [Phase::Cold,Phase::Sample(123),Phase::Coupled{parity:0,seed:9},Phase::History(3,0.125)] {
            let bytes=to_bytes(&value).unwrap();assert_eq!(from_bytes::<Phase>(&bytes).unwrap(),value);
            for n in 0..bytes.len(){assert!(from_bytes::<Phase>(&bytes[..n]).is_err());}
            let mut extra=bytes;extra.push(0);assert!(from_bytes::<Phase>(&extra).is_err());
        }
    }
    #[test]fn oversized_length_refused_before_allocation(){let mut bytes=MAGIC.to_vec();bytes.push(14);bytes.extend_from_slice(&u64::MAX.to_le_bytes());assert!(from_bytes::<Vec<u8>>(&bytes).is_err());}
}

/// Explicit raw-byte Serde view for large native planes. Prevents representing
/// every byte as a tagged u64 when nesting device capsules in owner bundles.
pub struct ByteView<'a>(pub &'a [u8]);
const OWNED_BYTE_VIEW_ALLOCATION:&str="prism.owned-byte-view-allocation.v1";
struct BytePayload<'a>(&'a [u8]);
impl Serialize for BytePayload<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{s.serialize_bytes(self.0)}}
impl Serialize for ByteView<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{
    s.serialize_newtype_struct(OWNED_BYTE_VIEW_ALLOCATION,&BytePayload(self.0))
}}
pub struct ByteBuffer(pub Vec<u8>);
impl<'de> Deserialize<'de> for ByteBuffer {
    fn deserialize<D:serde::Deserializer<'de>>(d:D)->std::result::Result<Self,D::Error>{
        struct V;impl<'de> Visitor<'de> for V {type Value=ByteBuffer;
            fn expecting(&self,f:&mut fmt::Formatter)->fmt::Result{f.write_str("an exact native byte plane")}
            fn visit_bytes<E:serde::de::Error>(self,v:&[u8])->std::result::Result<Self::Value,E>{Ok(ByteBuffer(v.to_vec()))}
            fn visit_byte_buf<E:serde::de::Error>(self,v:Vec<u8>)->std::result::Result<Self::Value,E>{Ok(ByteBuffer(v))}
            fn visit_seq<A:SeqAccess<'de>>(self,mut a:A)->std::result::Result<Self::Value,A::Error>{
                let mut v=Vec::new();while let Some(b)=a.next_element::<u8>()?{v.push(b);}Ok(ByteBuffer(v))}
        }d.deserialize_byte_buf(V)
    }
}
pub mod bytes {
    use super::*;
    pub fn serialize<S:serde::Serializer>(value:&[u8],s:S)->std::result::Result<S::Ok,S::Error>{s.serialize_bytes(value)}
    pub fn deserialize<'de,D:serde::Deserializer<'de>>(d:D)->std::result::Result<Vec<u8>,D::Error>{ByteBuffer::deserialize(d).map(|v|v.0)}
}
pub mod optional_bytes {
    use super::*;
    pub fn serialize<S:serde::Serializer>(value:&Option<Vec<u8>>,s:S)->std::result::Result<S::Ok,S::Error>{
        value.as_ref().map(|v|ByteView(v)).serialize(s)}
    pub fn deserialize<'de,D:serde::Deserializer<'de>>(d:D)->std::result::Result<Option<Vec<u8>>,D::Error>{
        Option::<ByteBuffer>::deserialize(d).map(|v|v.map(|v|v.0))}
}
pub mod byte_map {
    use super::*;use std::collections::BTreeMap;
    pub fn serialize<S:serde::Serializer>(value:&BTreeMap<String,Vec<u8>>,s:S)->std::result::Result<S::Ok,S::Error>{
        let mut map=s.serialize_map(Some(value.len()))?;for(k,v)in value{map.serialize_entry(k,&ByteView(v))?;}map.end()}
    pub fn deserialize<'de,D:serde::Deserializer<'de>>(d:D)->std::result::Result<BTreeMap<String,Vec<u8>>,D::Error>{
        BTreeMap::<String,ByteBuffer>::deserialize(d).map(|v|v.into_iter().map(|(k,v)|(k,v.0)).collect())}
}
#[cfg(test)]
mod byte_plane_tests {
    use super::*;
    #[derive(Serialize,Deserialize)]struct Plane {#[serde(with="bytes")] plane:Vec<u8>,#[serde(with="optional_bytes")] optional:Option<Vec<u8>>,#[serde(with="byte_map")] map:std::collections::BTreeMap<String,Vec<u8>>}
    #[test]fn native_planes_stay_linear_and_roundtrip_exact(){
        let bytes=(0..1048576).map(|n|(n%251)as u8).collect::<Vec<_>>();
        let value=Plane{plane:bytes.clone(),optional:Some(bytes.clone()),map:std::collections::BTreeMap::from([("raw".into(),bytes.clone())])};
        let wire=to_bytes(&value).unwrap();assert!(wire.len()<bytes.len()*3+256);
        let restored:Plane=from_bytes(&wire).unwrap();assert_eq!(restored.plane,bytes);assert_eq!(restored.optional.as_ref(),Some(&bytes));assert_eq!(restored.map["raw"],bytes);
    }
}
