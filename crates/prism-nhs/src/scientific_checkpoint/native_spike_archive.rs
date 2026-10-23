//! Explicit lossless native host-vector projection onto already committed raw
//! Chronicle objects. Original96-byte field bits and vector ordering survive;
//! no Rust padding, event filtering, JSON float conversion or data re-emission.
use super::*;
use super::chronicle::{PayloadReference,ProducerKey};
use super::lossless_object::LosslessObjectEnvelope;
use super::restart_transaction::StagedField;
use std::os::unix::fs::{MetadataExt,FileExt};
pub const NATIVE_SPIKE_ARCHIVE_ENCODING:&str="prism.native-spike-vector96.v1";
const ARCHIVE_CODEC:&str="prism.chronicle-spike-archive.v1";
const RECORD_BYTES:u64=96;
const MAX_RAW_LEAF_BYTES:u64=4*1024*1024;
#[derive(Clone,PartialEq,Eq)]
pub(super) struct RawStamp {dev:u64,inode:u64,len:u64,mtime:i64,mtime_ns:i64,ctime:i64,ctime_ns:i64}
fn raw_stamp(path:&Path)->Result<RawStamp> {
    let m=fs::symlink_metadata(path)?;ensure!(m.is_file()&&!m.file_type().is_symlink(),"archive raw source is not regular");
    Ok(RawStamp{dev:m.dev(),inode:m.ino(),len:m.len(),mtime:m.mtime(),mtime_ns:m.mtime_nsec(),ctime:m.ctime(),ctime_ns:m.ctime_nsec()})
}
/// Opaque same-process optimization witness. It cannot be deserialized or
/// supplied by a capsule. A fresh process, native clear/reset/restore, owner
/// replacement, changed prefix or changed inode stamp forces fresh comparison.
#[derive(Clone)]
pub struct NativeSpikeArchiveWitness {
    root:PathBuf,identity:CheckpointIdentity,producer:ProducerKey,seed:u64,
    instance:u64,mutation_epoch:u64,first:u64,next:u64,
    payloads:Vec<PayloadReference>,stamps:Vec<RawStamp>,hash:Sha256,
}
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeSpikeArchiveV1 {
    schema_version:u32,contract:FieldContract,identity:CheckpointIdentity,
    input_foliage:Option<Head>,producer:ProducerKey,replica_seed:u64,
    first_ordinal:u64,next_ordinal:u64,record_bytes:u64,payloads:Vec<PayloadReference>,
}
impl CheckpointStore {
    /// The callback compares these actual hash-checked original raw bytes with
    /// the corresponding canonical96B native host rows. Only the newly appended
    /// suffix is compared when the opaque append-only witness remains valid.
    pub fn stage_native_spike_archive(&self,contract:&FieldContract,identity:&CheckpointIdentity,
        head:Option<&Head>,producer:ProducerKey,seed:u64,instance:u64,mutation_epoch:u64,
        first:u64,next:u64,previous:Option<&NativeSpikeArchiveWitness>,
        mut compare_native:impl FnMut(u64,&[u8])->Result<()>)->Result<(StagedField,NativeSpikeArchiveWitness)> {
        contract.validate()?;identity.validate()?;
        ensure!(contract.name=="accumulated_spikes" && contract.encoding==NATIVE_SPIKE_ARCHIVE_ENCODING && !contract.may_be_absent
            && producer.field_id==0 && producer.stream_id<identity.stream_count && first<=next,"native archive field/producer/range differs");
        ensure!(contract.owner==format!("stream_{}/engine/fused",producer.stream_id),"native archive escaped its exact engine owner");
        let _lock=self.scientific_staging_guard()?;self.ensure_scientific_producer_open()?;ensure!(self.chronicle_head()?.as_ref()==head,"native archive source HEAD changed");
        let payloads=self.native_archive_payloads(identity,head,&producer,first,next)?;
        let stamps=payloads.iter().map(|reference|raw_stamp(&self.root.join("objects").join(&reference.object_sha256))).collect::<Result<Vec<_>>>()?;
        let reusable=previous.filter(|prior|prior.root==self.root && prior.identity==*identity && prior.producer==producer
            && prior.seed==seed && prior.instance==instance && prior.mutation_epoch==mutation_epoch && prior.first==first
            && prior.next<=next && prior.payloads.len()<=payloads.len()
            && prior.payloads.as_slice()==&payloads[..prior.payloads.len()]
            && prior.stamps.as_slice()==&stamps[..prior.payloads.len()]);
        let mut hash=reusable.map_or_else(Sha256::new,|prior|prior.hash.clone());
        let mut frontier=reusable.map_or(first,|prior|prior.next);
        for reference in &payloads {
            if reference.coverage.next_ordinal<=frontier{continue;}
            let from=reference.coverage.first_ordinal.max(frontier);let end=reference.coverage.next_ordinal.min(next);
            ensure!(from==frontier && end>from,"native archive new suffix has an ordinal gap");
            let length=(reference.coverage.next_ordinal-reference.coverage.first_ordinal).checked_mul(RECORD_BYTES).context("native raw extent overflow")?;
            ensure!(length<=MAX_RAW_LEAF_BYTES,"native raw archive source exceeds bounded slab");
            let bytes=read_bounded(&self.root.join("objects").join(&reference.object_sha256),length)?;
            ensure!(bytes.len() as u64==length && super::hash(&bytes)==reference.object_sha256,"native archive source bytes changed");
            let offset=usize::try_from((from-reference.coverage.first_ordinal)*RECORD_BYTES)?;
            let extent=usize::try_from((end-from)*RECORD_BYTES)?;let suffix=&bytes[offset..offset+extent];
            compare_native(from-first,suffix)?;hash.update(suffix);frontier=end;
        }
        ensure!(frontier==next,"native archive suffix population incomplete");
        let decoded_bytes=(next-first).checked_mul(RECORD_BYTES).context("native archive decoded extent overflow")?;
        ensure!(decoded_bytes<=self.max_total_bytes,"native archive exceeds decoded field admission");
        let manifest=NativeSpikeArchiveV1{schema_version:1,contract:contract.clone(),identity:identity.clone(),input_foliage:head.cloned(),
            producer:producer.clone(),replica_seed:seed,first_ordinal:first,next_ordinal:next,record_bytes:RECORD_BYTES,payloads:payloads.clone()};
        let metadata=serde_json::to_vec(&manifest)?;ensure!(metadata.len() as u64<=MAX_MANIFEST_BYTES,"native archive reference manifest exceeds bounded admission");
        let envelope=LosslessObjectEnvelope{schema_version:1,codec:ARCHIVE_CODEC.into(),encoder_version:"native96-reference-v1".into(),window_log:0,
            encoded_sha256:super::hash(&metadata),encoded_bytes:metadata.len() as u64,decoded_sha256:format!("{:x}",hash.clone().finalize()),decoded_bytes};
        let path=self.root.join("encoded_objects").join(&envelope.encoded_sha256);
        if path.try_exists()? {self.verify_encoded_file(&envelope)?;}else{
            self.spend_reserved_capacity(metadata.len() as u64)?;write_new(&path,&metadata)?;sync_dir(&self.root.join("encoded_objects"))?;
        }
        // Full old-prefix bytes were previously compared under the opaque
        // witness; every new byte was compared above and hashed here. Restore
        // always reads/hash-checks the actual complete referenced population.
        let stamp=self.lossless_file_stamp(&envelope)?;
        let mut verified=self.verified_lossless.lock().map_err(|_|anyhow::anyhow!("native archive witness poisoned"))?;
        if verified.len()>=32768{verified.clear();}
        verified.insert(envelope.encoded_sha256.clone(),(envelope.clone(),stamp));drop(verified);
        let budget=decoded_bytes.checked_mul(4).and_then(|n|n.checked_add(128)).context("native vector heap budget overflow")?;
        self.remember_native_archive_sources(&envelope.encoded_sha256,payloads.iter().zip(&stamps).map(|(reference,stamp)|(reference.object_sha256.clone(),stamp.clone())).collect())?;
        let staged=StagedField{store_root:self.root.clone(),stored:StoredField{contract:contract.clone(),present:true,byte_count:decoded_bytes,
            object_sha256:Some(envelope.decoded_sha256.clone()),representation:Some(envelope),host_allocation_budget:Some(budget)}};
        Ok((staged,NativeSpikeArchiveWitness{root:self.root.clone(),identity:identity.clone(),producer,seed,instance,mutation_epoch,first,next,payloads,stamps,hash}))
    }
    fn native_archive_payloads(&self,identity:&CheckpointIdentity,head:Option<&Head>,producer:&ProducerKey,first:u64,next:u64)->Result<Vec<PayloadReference>> {
        let Some(head)=head else{ensure!(first==0&&next==0,"native event archive lacks raw HEAD");return Ok(Vec::new());};
        let foliage=self.read_foliage(head)?;ensure!(foliage.identity==*identity,"native archive input execution identity changed");
        let mut payloads=Vec::new();let mut frontier=first;
        for leaf in foliage.leaves.iter().filter(|leaf|leaf.producer==*producer && leaf.coverage.next_ordinal>first && leaf.coverage.first_ordinal<next) {
            let start=leaf.coverage.first_ordinal.max(first);let end=leaf.coverage.next_ordinal.min(next);
            ensure!(start==frontier && end>start && leaf.byte_count==(leaf.coverage.next_ordinal-leaf.coverage.first_ordinal)*RECORD_BYTES
                && leaf.byte_count<=MAX_RAW_LEAF_BYTES,"native archive reference range/layout differs");
            payloads.push(leaf.reference());ensure!(payloads.len()<=32768,"native archive reference population exceeds bounded admission");frontier=end;
        }ensure!(frontier==next,"native archive lacks complete original ordinal population");Ok(payloads)
    }
    fn read_native_spike_archive(&self,envelope:&LosslessObjectEnvelope)->Result<NativeSpikeArchiveV1> {
        ensure!(envelope.schema_version==1 && envelope.codec==ARCHIVE_CODEC && envelope.window_log==0
            && envelope.encoded_bytes<=MAX_MANIFEST_BYTES,"unsupported native spike archive envelope");
        self.verify_encoded_file(envelope)?;
        let manifest:NativeSpikeArchiveV1=serde_json::from_slice(&read_bounded(&self.root.join("encoded_objects").join(&envelope.encoded_sha256),MAX_MANIFEST_BYTES)?)?;
        manifest.contract.validate()?;manifest.identity.validate()?;
        ensure!(manifest.schema_version==1 && manifest.record_bytes==RECORD_BYTES && manifest.producer.field_id==0
            && manifest.producer.stream_id<manifest.identity.stream_count && manifest.first_ordinal<=manifest.next_ordinal
            && manifest.contract.name=="accumulated_spikes" && manifest.contract.encoding==NATIVE_SPIKE_ARCHIVE_ENCODING
            && manifest.contract.owner==format!("stream_{}/engine/fused",manifest.producer.stream_id)
            && (manifest.next_ordinal-manifest.first_ordinal).checked_mul(RECORD_BYTES)==Some(envelope.decoded_bytes),"native spike archive identity/layout differs");
        ensure!(manifest.payloads==self.native_archive_payloads(&manifest.identity,manifest.input_foliage.as_ref(),&manifest.producer,manifest.first_ordinal,manifest.next_ordinal)?,
            "native event archive is not an exact member population of its committed input foliage");Ok(manifest)
    }
    fn remember_native_archive_sources(&self,key:&str,stamps:Vec<(String,RawStamp)>)->Result<()> {
        ensure!(stamps.len()<=32768,"native archive source witness exceeds field bound");
        let mut cache=self.verified_native_archive_sources.lock().map_err(|_|anyhow::anyhow!("native archive source cache poisoned"))?;
        let total=cache.values().try_fold(stamps.len(),|n,v|n.checked_add(v.len()).context("source cache count overflow"))?;
        if total>65536 || cache.len()>=32 {cache.clear();}
        cache.insert(key.into(),stamps);Ok(())
    }
    pub(super) fn verify_native_archive_members(&self,envelope:&LosslessObjectEnvelope)->Result<()> {
        let manifest=self.read_native_spike_archive(envelope)?;
        let current=manifest.payloads.iter().map(|reference|Ok((reference.object_sha256.clone(),raw_stamp(&self.root.join("objects").join(&reference.object_sha256))?))).collect::<Result<Vec<_>>>()?;
        if self.verified_native_archive_sources.lock().map_err(|_|anyhow::anyhow!("native source cache poisoned"))?
            .get(&envelope.encoded_sha256).is_some_and(|saved|saved==&current) {return Ok(());}
        for reference in manifest.payloads {
            let length=(reference.coverage.next_ordinal-reference.coverage.first_ordinal)*RECORD_BYTES;
            self.verify_object_streaming(&reference.object_sha256,length)?;
        }
        self.remember_native_archive_sources(&envelope.encoded_sha256,current)
    }
    pub(super) fn open_native_archive_reader(&self,envelope:&LosslessObjectEnvelope)->Result<Box<dyn Read>> {
        let manifest=self.read_native_spike_archive(envelope)?;
        self.verify_native_archive_members(envelope)?;
        Ok(Box::new(NativeArchiveReader{root:self.root.clone(),first:manifest.first_ordinal,next:manifest.next_ordinal,
            payloads:manifest.payloads.into_iter(),current:None}))
    }
    pub(super) fn link_native_archive_closure(&self,target:&CheckpointStore,envelope:&LosslessObjectEnvelope)->Result<()> {
        let manifest=self.read_native_spike_archive(envelope)?;
        for reference in manifest.payloads {self.verify_object_streaming(&reference.object_sha256,
            (reference.coverage.next_ordinal-reference.coverage.first_ordinal)*RECORD_BYTES)?;
            self.link_immutable_to(target,"objects",&reference.object_sha256)?;}
        // The full RawChronicle HEAD ancestry is forked by the existing raw
        // owner in the same ingress before native restore may inspect it.
        Ok(())
    }
    pub(super) fn accumulate_native_archive_branch_bound(&self,envelope:&LosslessObjectEnvelope,objects:&mut BTreeSet<(String,String)>,bound:&mut u64)->Result<()> {
        for reference in self.read_native_spike_archive(envelope)?.payloads {
            if objects.insert(("objects".into(),reference.object_sha256)) {*bound=bound.checked_add(super::restart_transaction::branch_copy_member_bound(0)?).context("native archive link bound overflow")?;}
            ensure!(objects.len()<=262144,"native archive branch metadata population exceeded");
        }Ok(())
    }
}
struct NativeArchiveReader {root:PathBuf,first:u64,next:u64,payloads:std::vec::IntoIter<PayloadReference>,current:Option<std::io::Take<std::io::BufReader<File>>>}
impl Read for NativeArchiveReader {
    fn read(&mut self,bytes:&mut[u8])->std::io::Result<usize> {
        use std::io::{Seek,SeekFrom};
        if bytes.is_empty(){return Ok(0);}
        loop {
            if let Some(input)=self.current.as_mut() {let n=input.read(bytes)?;if n>0{return Ok(n);}if input.limit()!=0{return Err(std::io::Error::other("native archive source truncated"));}self.current=None;}
            let Some(reference)=self.payloads.next()else{return Ok(0);};
            let first=reference.coverage.first_ordinal.max(self.first);let next=reference.coverage.next_ordinal.min(self.next);
            let mut file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(self.root.join("objects").join(reference.object_sha256))?;
            file.seek(SeekFrom::Start((first-reference.coverage.first_ordinal)*RECORD_BYTES))?;self.current=Some(std::io::BufReader::with_capacity(65536,file).take((next-first)*RECORD_BYTES));
        }
    }
}
#[cfg(feature="gpu")]
impl super::restart_transaction::ScientificGenerationReader<'_> {
    pub fn native_spikes(&self,owner:&str,name:&str)->Result<Vec<crate::fused_engine::GpuSpikeEvent>> {
        ensure!(self.index.belongs_to_store(self.store),"native vector reader store/generation admission differs");
        let field=self.index.native_archive_field(owner,name)?;
        ensure!(field.contract.encoding==NATIVE_SPIKE_ARCHIVE_ENCODING && field.present,"native host-vector field encoding/presence differs");
        let envelope=field.representation.as_ref().context("native vector lacks archive representation")?;
        if envelope.codec==ARCHIVE_CODEC {
            let manifest=self.store.read_native_spike_archive(envelope)?;
            ensure!(manifest.contract==field.contract && manifest.identity==*self.index.identity()
                && self.index.boundaries().iter().any(|b|b.stream_id==manifest.producer.stream_id && b.replica_epoch==manifest.producer.replica_epoch && b.replica_seed==manifest.replica_seed),
                "native archive field/provider/replica identity differs from scientific generation");
        }else{ensure!(matches!(envelope.codec.as_str(),"zstd.frame.v1"|"zstd.chunked-field.v1"),"unsupported explicit native96 field representation");}
        ensure!(envelope.decoded_bytes==field.byte_count && Some(&envelope.decoded_sha256)==field.object_sha256.as_ref(),"native archive decoded commitment differs");
        ensure!(envelope.decoded_bytes%RECORD_BYTES==0,"native vector contains an incomplete record");
        let count=usize::try_from(envelope.decoded_bytes/RECORD_BYTES)?;
        let required=(count as u64).checked_mul(std::mem::size_of::<crate::fused_engine::GpuSpikeEvent>() as u64).context("native vector restore allocation overflow")?;
        ensure!(required.checked_mul(4).and_then(|n|n.checked_add(128)).zip(field.host_allocation_budget).is_some_and(|(need,budget)|need<=budget),"native event vector exceeds pre-admitted heap allocation");
        let mut events=Vec::new();events.try_reserve_exact(count)?;
        let mut input=self.store.open_lossless_reader(envelope)?;let mut hash=Sha256::new();let mut record=[0u8;96];
        for _ in 0..count {input.read_exact(&mut record)?;hash.update(record);events.push(crate::fused_engine::GpuSpikeEvent::from_le_bytes_with_stride(&record,96)?);}
        let mut extra=[0u8;1];ensure!(input.read(&mut extra)?==0 && format!("{:x}",hash.finalize())==envelope.decoded_sha256,"restored native vector population/hash differs");Ok(events)
    }
}
