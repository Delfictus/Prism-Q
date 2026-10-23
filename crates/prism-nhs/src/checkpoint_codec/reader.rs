//! Owned decoding of the same typed bytes without a complete wire-image Vec.
use super::*;
use std::io::Read;
use serde::de::{DeserializeOwned,Error as _};
pub fn from_reader<T:DeserializeOwned>(reader:&mut dyn Read,wire_bytes:u64,max_fragment_bytes:u64)->Result<T> {
    from_reader_with_allocation_budget(reader,wire_bytes,max_fragment_bytes,u64::MAX)
}
pub fn from_reader_with_allocation_budget<T:DeserializeOwned>(reader:&mut dyn Read,wire_bytes:u64,max_fragment_bytes:u64,heap_budget:u64)->Result<T> {
    let mut d=ReaderDecoder{reader,remaining:wire_bytes,max_fragment:max_fragment_bytes,depth:0,heap_remaining:heap_budget,heap_budget,wire_bytes,target_type:std::any::type_name::<T>()};
    d.charge((std::mem::size_of::<T>() as u64).checked_mul(4).ok_or_else(||err("host inline bound overflow"))?)?;
    if d.word::<8>()?!=*MAGIC{return Err(err("invalid scientific host-state stream"));}
    let value=T::deserialize(&mut d)?;
    if d.remaining!=0{return Err(err("trailing scientific host-state stream"));}Ok(value)
}
struct ReaderDecoder<'r>{reader:&'r mut dyn Read,remaining:u64,max_fragment:u64,depth:usize,heap_remaining:u64,heap_budget:u64,wire_bytes:u64,target_type:&'static str}
impl ReaderDecoder<'_> {
    fn charge(&mut self,bytes:u64)->Result<()> {
        self.heap_remaining=self.heap_remaining.checked_sub(bytes).ok_or_else(||err(&format!(
            "host allocation exceeds pre-admitted typed budget: type={} wire_bytes={} wire_remaining={} planned_budget={} requested_charge={} remaining_budget={} depth={}",
            self.target_type,self.wire_bytes,self.remaining,self.heap_budget,bytes,self.heap_remaining,self.depth)))?;Ok(())
    }
    fn read(&mut self,bytes:&mut[u8])->Result<()> {
        if bytes.len() as u64>self.remaining{return Err(err("truncated scientific host-state stream"));}
        self.reader.read_exact(bytes).map_err(Error::custom)?;self.remaining-=bytes.len() as u64;Ok(())
    }
    fn word<const N:usize>(&mut self)->Result<[u8;N]>{let mut out=[0;N];self.read(&mut out)?;Ok(out)}
    fn tag(&mut self)->Result<u8>{Ok(self.word::<1>()?[0])}
    fn length(&mut self)->Result<usize>{let n=u64::from_le_bytes(self.word::<8>()?);if n>self.remaining{return Err(err("host-state count exceeds remaining stream"));}usize::try_from(n).map_err(Error::custom)}
    fn take(&mut self,n:usize)->Result<Vec<u8>> {
        if n as u64>self.max_fragment{return Err(err("single owned host fragment exceeds admitted allocation"));}
        self.charge((n as u64).checked_mul(4).and_then(|n|n.checked_add(64)).ok_or_else(||err("owned fragment bound overflow"))?)?;
        let mut bytes=Vec::new();bytes.try_reserve_exact(n).map_err(Error::custom)?;bytes.resize(n,0);self.read(&mut bytes)?;Ok(bytes)
    }
    fn string(&mut self)->Result<String>{if self.tag()?!=10{return Err(err("expected scientific string"));}let n=self.length()?;String::from_utf8(self.take(n)?).map_err(Error::custom)}
}
impl<'de> serde::Deserializer<'de> for &mut ReaderDecoder<'_>{
    type Error=Error;
    fn deserialize_any<V:Visitor<'de>>(self,v:V)->Result<V::Value>{
        if self.depth>=MAX_DEPTH{return Err(err("host-state nesting limit"));}self.depth+=1;
        let out=match self.tag()? {
            0=>v.visit_unit(),1=>v.visit_bool(false),2=>v.visit_bool(true),
            3=>v.visit_i64(i64::from_le_bytes(self.word::<8>()?)),4=>v.visit_u64(u64::from_le_bytes(self.word::<8>()?)),
            5=>v.visit_i128(i128::from_le_bytes(self.word::<16>()?)),6=>v.visit_u128(u128::from_le_bytes(self.word::<16>()?)),
            7=>v.visit_f32(f32::from_bits(u32::from_le_bytes(self.word::<4>()?))),8=>v.visit_f64(f64::from_bits(u64::from_le_bytes(self.word::<8>()?))),
            9=>v.visit_char(char::from_u32(u32::from_le_bytes(self.word::<4>()?)).ok_or_else(||err("invalid state character"))?),
            10=>{let n=self.length()?;v.visit_string(String::from_utf8(self.take(n)?).map_err(Error::custom)?)},
            11=>{let n=self.length()?;v.visit_byte_buf(self.take(n)?)},
            12=>v.visit_none(),13=>v.visit_some(&mut*self),
            14=>{self.charge(64)?;let n=self.length()?;let mut seq=ReaderSequence{d:self,remaining:n};let out=v.visit_seq(&mut seq)?;if seq.remaining!=0{return Err(err("unconsumed sequence state"));}Ok(out)},
            15=>{self.charge(64)?;let n=self.length()?;let mut map=ReaderMapping{d:self,remaining:n,pending:false,charge_elements:true};let out=v.visit_map(&mut map)?;if map.remaining!=0||map.pending{return Err(err("unconsumed map state"));}Ok(out)},
            16=>{let variant=self.string()?;v.visit_enum(ReaderEnumeration{d:self,variant})},
            _=>Err(err("unknown scientific state tag")),
        };self.depth-=1;out
    }
    fn deserialize_option<V:Visitor<'de>>(self,v:V)->Result<V::Value>{match self.tag()?{12=>v.visit_none(),13=>v.visit_some(self),_=>Err(err("expected optional state tag"))}}
    fn deserialize_newtype_struct<V:Visitor<'de>>(self,_:&'static str,v:V)->Result<V::Value>{v.visit_newtype_struct(self)}
    fn deserialize_enum<V:Visitor<'de>>(self,_:&'static str,_:&'static[&'static str],v:V)->Result<V::Value>{if self.tag()?!=16{return Err(err("expected enum state tag"));}let variant=self.string()?;v.visit_enum(ReaderEnumeration{d:self,variant})}
    fn deserialize_struct<V:Visitor<'de>>(self,_:&'static str,_:&'static[&'static str],v:V)->Result<V::Value> {
        if self.depth>=MAX_DEPTH{return Err(err("host-state nesting limit"));}
        if self.tag()?!=15{return Err(err("expected scientific struct map"));}
        self.charge(64)?;self.depth+=1;let n=self.length()?;
        let mut map=ReaderMapping{d:self,remaining:n,pending:false,charge_elements:false};
        let out=v.visit_map(&mut map)?;
        if map.remaining!=0||map.pending{return Err(err("unconsumed scientific struct state"));}
        map.d.depth-=1;Ok(out)
    }
    fn is_human_readable(&self)->bool{false}
    serde::forward_to_deserialize_any!{bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf unit unit_struct seq tuple tuple_struct map identifier ignored_any}
}
struct ReaderSequence<'a,'r>{d:&'a mut ReaderDecoder<'r>,remaining:usize}
impl<'de> SeqAccess<'de> for &mut ReaderSequence<'_,'_>{type Error=Error;
    fn next_element_seed<T:DeserializeSeed<'de>>(&mut self,seed:T)->Result<Option<T::Value>>{if self.remaining==0{return Ok(None);}self.d.charge((std::mem::size_of::<T::Value>() as u64).checked_mul(4).and_then(|n|n.checked_add(16)).ok_or_else(||err("sequence heap bound overflow"))?)?;self.remaining-=1;Ok(Some(seed.deserialize(&mut*self.d)?))}
    fn size_hint(&self)->Option<usize>{None}}
struct ReaderMapping<'a,'r>{d:&'a mut ReaderDecoder<'r>,remaining:usize,pending:bool,charge_elements:bool}
impl<'de> MapAccess<'de> for &mut ReaderMapping<'_,'_>{type Error=Error;
    fn next_key_seed<T:DeserializeSeed<'de>>(&mut self,seed:T)->Result<Option<T::Value>>{if self.pending{return Err(err("map value not consumed"));}if self.remaining==0{return Ok(None);}if self.charge_elements {self.d.charge((std::mem::size_of::<T::Value>() as u64).checked_mul(4).and_then(|n|n.checked_add(256)).ok_or_else(||err("map key heap bound overflow"))?)?;}self.pending=true;Ok(Some(seed.deserialize(&mut*self.d)?))}
    fn next_value_seed<T:DeserializeSeed<'de>>(&mut self,seed:T)->Result<T::Value>{if !self.pending{return Err(err("map value lacks key"));}if self.charge_elements {self.d.charge((std::mem::size_of::<T::Value>() as u64).checked_mul(4).and_then(|n|n.checked_add(256)).ok_or_else(||err("map value heap bound overflow"))?)?;}let value=seed.deserialize(&mut*self.d)?;self.pending=false;self.remaining-=1;Ok(value)}
    fn size_hint(&self)->Option<usize>{None}}
struct ReaderEnumeration<'a,'r>{d:&'a mut ReaderDecoder<'r>,variant:String}
impl<'de> EnumAccess<'de> for ReaderEnumeration<'_,'_>{type Error=Error;type Variant=Self;
    fn variant_seed<T:DeserializeSeed<'de>>(self,seed:T)->Result<(T::Value,Self)>{Ok((seed.deserialize(self.variant.clone().into_deserializer())?,self))}}
impl<'de> VariantAccess<'de> for ReaderEnumeration<'_,'_>{type Error=Error;
    fn unit_variant(self)->Result<()>{Deserialize::deserialize(self.d)}
    fn newtype_variant_seed<T:DeserializeSeed<'de>>(self,seed:T)->Result<T::Value>{seed.deserialize(self.d)}
    fn tuple_variant<V:Visitor<'de>>(self,_:usize,v:V)->Result<V::Value>{serde::Deserializer::deserialize_seq(self.d,v)}
    fn struct_variant<V:Visitor<'de>>(self,_:&'static[&'static str],v:V)->Result<V::Value>{serde::Deserializer::deserialize_map(self.d,v)}}
