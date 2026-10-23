//! Exact sizing for the existing typed wire format, before payload allocation.
use super::*;
use serde::ser::Error as _;

pub fn encoded_size<T: Serialize + ?Sized>(value: &T, limit: u64) -> Result<u64> {
    let mut counter = Counter {
        bytes: 8,
        limit,
        depth: 0,heap_bytes:0,
    };
    counter.add(0)?;
    value.serialize(&mut counter)?;
    Ok(counter.bytes)
}

pub fn to_bytes_bounded<T: Serialize + ?Sized>(value: &T, limit: u64) -> Result<Vec<u8>> {
    let size = encoded_size(value, limit)?;
    let capacity = usize::try_from(size).map_err(Error::custom)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(Error::custom)?;
    bytes.extend_from_slice(MAGIC);
    let mut encoder = Encoder { bytes, depth: 0 };
    value.serialize(&mut encoder)?;
    if encoder.bytes.len() != capacity {
        return Err(err(
            "typed source mutated between size admission and serialization",
        ));
    }
    Ok(encoder.bytes)
}
/// Serde collect_seq/collect_map pass a reference-valued iterator item to
/// serialize_element/entry. The planner must charge the owned decode slot,
/// rather than the extra borrowed pointer used only by serialization.
/// These explicit standard owned shapes are shared across all host fields.
fn source_owned_inline_bytes<T:Serialize+?Sized>(value:&T)->u64 {
    let name=std::any::type_name::<T>();
    let view=std::mem::size_of_val(value) as u64;
    macro_rules! owned_shape {($($owned:ty),* $(,)?)=>{$(
        if name==std::any::type_name::<&$owned>() {return view.max(std::mem::size_of::<$owned>() as u64);}
    )*};}
    // Explicit current native owned element layouts, including nested JSON
    // arrays and the physical snapshot/persistence populations. Type names
    // select compile-time size_of facts in this exact engine build; unknown
    // Serialize implementations retain legacy accounting, not a universal
    // owned-layout certification.
    owned_shape!(serde_json::Value,String,ByteView<'_>,
        ([f32;3],i32),(f32,i32),[f32;3],[f64;3],(u64,u64),(i32,f64),(u64,f64));
    #[cfg(feature="gpu")]
    owned_shape!(crate::fused_engine::SpikeEvent,crate::fused_engine::EnsembleSnapshot,
        crate::fused_engine::SpikeQualityScore);
    if name.starts_with("&alloc::vec::Vec<") {return view.max(std::mem::size_of::<Vec<()>>() as u64);}
    if name.starts_with("&alloc::collections::vec_deque::VecDeque<") {return view.max(std::mem::size_of::<std::collections::VecDeque<()>>() as u64);}
    if name.starts_with("&alloc::collections::btree::map::BTreeMap<") {return view.max(std::mem::size_of::<std::collections::BTreeMap<(),()>>() as u64);}
    if name.starts_with("&alloc::collections::btree::set::BTreeSet<") {return view.max(std::mem::size_of::<std::collections::BTreeSet<()>>() as u64);}
    view
}

struct Counter {
    bytes: u64,
    limit: u64,
    depth: usize,
    heap_bytes:u64,
}
/// Counts the native serde population without allocating its wire image. The
/// heap budget deliberately includes growing-container and map-node headroom;
/// the streamed decoder charges the same typed operations before allocation.
pub fn allocation_plan<T:Serialize+?Sized>(value:&T,limit:u64)->Result<(u64,u64)> {
    let mut counter=Counter{bytes:8,limit,depth:0,heap_bytes:0};
    counter.heap_add((std::mem::size_of_val(value) as u64).checked_mul(4).ok_or_else(||err("host object size overflow"))?)?;
    value.serialize(&mut counter)?;Ok((counter.bytes,counter.heap_bytes))
}
impl Counter {
    fn heap_add(&mut self,bytes:u64)->Result<()> {
        self.heap_bytes=self.heap_bytes.checked_add(bytes).ok_or_else(||err("host allocation accounting overflow"))?;Ok(())
    }
    fn add(&mut self, bytes: u64) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| err("typed wire size overflow"))?;
        if self.bytes > self.limit {
            return Err(err("typed wire size exceeds preallocation admission"));
        }
        Ok(())
    }
    fn string(&mut self, value: &str) -> Result<()> {
        self.heap_add((value.len() as u64).checked_mul(4).and_then(|n|n.checked_add(64)).ok_or_else(||err("string heap bound overflow"))?)?;
        self.add(9)?;
        self.add(value.len() as u64)
    }
    fn variant(&mut self, value: &str) -> Result<()> {
        self.add(1)?;
        self.string(value)
    }
    fn compound(&mut self, expected: Option<usize>,heap_kind:u8) -> Result<CountCompound<'_>> {
        if self.depth >= MAX_DEPTH {
            return Err(err("host-state nesting limit"));
        }
        self.heap_add(64)?;
        self.add(9)?;
        self.depth += 1;
        Ok(CountCompound {
            s: self,
            expected,
            count: 0,
            key_pending: false,heap_kind,
        })
    }
}
struct CountCompound<'a> {
    s: &'a mut Counter,
    expected: Option<usize>,
    count: usize,
    key_pending: bool,
    heap_kind:u8,
}
impl CountCompound<'_> {
    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<()> {
        if self.heap_kind!=0 {
            let inline=source_owned_inline_bytes(value).checked_mul(4).ok_or_else(||err("container inline bound overflow"))?;
            self.s.heap_add(inline.checked_add(if self.heap_kind==2{256}else{16}).ok_or_else(||err("container heap bound overflow"))?)?;
        }
        value.serialize(&mut *self.s)?;
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| err("collection count overflow"))?;
        Ok(())
    }
    fn field<T: Serialize + ?Sized>(&mut self, name: &str, value: &T) -> Result<()> {
        if self.heap_kind==2 {self.s.heap_add(512)?;}
        self.s.string(name)?;
        self.element(value)
    }
    fn finish(self) -> Result<()> {
        if self.key_pending || self.expected.is_some_and(|v| v != self.count) {
            return Err(err("serialized collection length mismatch"));
        }
        self.s.depth -= 1;
        Ok(())
    }
}
macro_rules! scalar_count {($($name:ident:$kind:ty=>$bytes:expr),*)=>{$(fn $name(self,_value:$kind)->Result<()> {self.add($bytes)})*}}
impl<'a> serde::Serializer for &'a mut Counter {
    type Ok = ();
    type Error = Error;
    type SerializeSeq = CountCompound<'a>;
    type SerializeTuple = CountCompound<'a>;
    type SerializeTupleStruct = CountCompound<'a>;
    type SerializeTupleVariant = CountCompound<'a>;
    type SerializeMap = CountCompound<'a>;
    type SerializeStruct = CountCompound<'a>;
    type SerializeStructVariant = CountCompound<'a>;
    scalar_count!(serialize_bool:bool=>1,serialize_i8:i8=>9,serialize_i16:i16=>9,serialize_i32:i32=>9,serialize_i64:i64=>9,
        serialize_u8:u8=>9,serialize_u16:u16=>9,serialize_u32:u32=>9,serialize_u64:u64=>9,
        serialize_i128:i128=>17,serialize_u128:u128=>17,serialize_f32:f32=>5,serialize_f64:f64=>9,serialize_char:char=>5);
    fn serialize_str(self, value: &str) -> Result<()> {
        self.string(value)
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<()> {
        self.heap_add((value.len() as u64).checked_mul(4).and_then(|n|n.checked_add(64)).ok_or_else(||err("byte heap bound overflow"))?)?;
        self.add(9)?;
        self.add(value.len() as u64)
    }
    fn serialize_none(self) -> Result<()> {
        self.add(1)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<()> {
        self.add(1)?;
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<()> {
        self.add(1)
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.add(1)
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.variant(variant)?;
        self.add(1)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<()> {
        if name==OWNED_BYTE_VIEW_ALLOCATION {
            // A byte-map serializes a borrowed ByteView but restores an owned
            // ByteBuffer. Container inline charges must include the actual Vec
            // header, not just the smaller borrowed slice header (32B at64bit).
            let extra=std::mem::size_of::<ByteBuffer>().checked_sub(std::mem::size_of::<ByteView<'_>>())
                .ok_or_else(||err("owned byte-view layout bound differs"))?;
            self.heap_add((extra as u64).checked_mul(4).ok_or_else(||err("owned byte-view heap bound overflow"))?)?;
        }
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<()> {
        self.variant(variant)?;
        value.serialize(self)
    }
    fn serialize_seq(self, length: Option<usize>) -> Result<Self::SerializeSeq> {
        self.compound(length,1)
    }
    fn serialize_tuple(self, length: usize) -> Result<Self::SerializeTuple> {
        self.compound(Some(length),1)
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        length: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.compound(Some(length),1)
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        length: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        self.variant(variant)?;
        self.compound(Some(length),1)
    }
    fn serialize_map(self, length: Option<usize>) -> Result<Self::SerializeMap> {
        self.compound(length,2)
    }
    fn serialize_struct(self, _name: &'static str, length: usize) -> Result<Self::SerializeStruct> {
        self.compound(Some(length),0)
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        length: usize,
    ) -> Result<Self::SerializeStructVariant> {
        self.variant(variant)?;
        self.compound(Some(length),2)
    }
    fn is_human_readable(&self) -> bool {
        false
    }
}
impl SerializeSeq for CountCompound<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        self.element(v)
    }
    fn end(self) -> Result<()> {
        self.finish()
    }
}
impl SerializeTuple for CountCompound<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        self.element(v)
    }
    fn end(self) -> Result<()> {
        self.finish()
    }
}
impl SerializeTupleStruct for CountCompound<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        self.element(v)
    }
    fn end(self) -> Result<()> {
        self.finish()
    }
}
impl SerializeTupleVariant for CountCompound<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        self.element(v)
    }
    fn end(self) -> Result<()> {
        self.finish()
    }
}
impl SerializeStruct for CountCompound<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, k: &'static str, v: &T) -> Result<()> {
        self.field(k, v)
    }
    fn end(self) -> Result<()> {
        self.finish()
    }
}
impl SerializeStructVariant for CountCompound<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, k: &'static str, v: &T) -> Result<()> {
        self.field(k, v)
    }
    fn end(self) -> Result<()> {
        self.finish()
    }
}
impl SerializeMap for CountCompound<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<()> {
        if self.key_pending {
            return Err(err("map key without value"));
        }
        self.s.heap_add(source_owned_inline_bytes(key).checked_mul(4).and_then(|n|n.checked_add(256)).ok_or_else(||err("map-key heap bound overflow"))?)?;
        key.serialize(&mut *self.s)?;
        self.key_pending = true;
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<()> {
        if !self.key_pending {
            return Err(err("map value without key"));
        }
        self.element(value)?;
        self.key_pending = false;
        Ok(())
    }
    fn end(self) -> Result<()> {
        self.finish()
    }
}
