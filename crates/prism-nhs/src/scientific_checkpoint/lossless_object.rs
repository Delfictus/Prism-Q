//! Explicit lossless representation for scientific field CAS objects.
//! Original FieldContract encodings describe decoded bytes without alteration.
//! Old raw objects/observational leaves keep their existing namespace/format.
use super::*;
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LosslessObjectEnvelope {
    pub schema_version:u32,
    pub codec:String,
    pub encoder_version:String,
    pub window_log:u32,
    pub encoded_sha256:String,
    pub encoded_bytes:u64,
    pub decoded_sha256:String,
    pub decoded_bytes:u64,
}
/// Bounded same-process immutable-object verification witness. The scientific
/// identity remains both SHA256 commitments in the envelope. A changed inode,
/// extent, mtime or ctime triggers full verification, never cache admission.
#[derive(Clone,Debug,PartialEq,Eq)]
pub(super) struct LosslessFileStamp {
    dev:u64,inode:u64,bytes:u64,mtime:i64,mtime_ns:i64,ctime:i64,ctime_ns:i64,
}
struct HashWriter {file:File,hasher:Sha256,count:u64,limit:u64}
impl Write for HashWriter {
    fn write(&mut self,bytes:&[u8])->std::io::Result<usize> {
        let next=self.count.checked_add(bytes.len() as u64).ok_or_else(||std::io::Error::new(std::io::ErrorKind::InvalidData,"compressed object size overflow"))?;
        if next>self.limit{return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,"compressed object exceeds admitted bound"));}
        let n=self.file.write(bytes)?;self.hasher.update(&bytes[..n]);self.count+=n as u64;Ok(n)
    }
    fn flush(&mut self)->std::io::Result<()>{self.file.flush()}
}
impl CheckpointStore {
    pub(super) fn encoded_index_path(&self,decoded:&str)->PathBuf {self.root.join("encoded_objects").join(format!("decoded_{decoded}.json"))}
    pub(super) fn encode_lossless_object(&self,bytes:&[u8])->Result<LosslessObjectEnvelope> {
        let leaf_start=std::time::Instant::now();let hash_start=std::time::Instant::now();
        let decoded=hash(bytes);let decoded_hash_ns=hash_start.elapsed().as_nanos();
        let lock_start=std::time::Instant::now();let _object=self.lossless_publication_guard(&decoded)?;let lock_wait_ns=lock_start.elapsed().as_nanos();let index=self.encoded_index_path(&decoded);
        if index.exists() {
            let envelope:LosslessObjectEnvelope=serde_json::from_slice(&read_bounded(&index,16384)?)?;
            ensure!(envelope.decoded_sha256==decoded && envelope.decoded_bytes==bytes.len() as u64,"lossless object index identity changed");
            let verify_start=std::time::Instant::now();self.verify_lossless_object(&envelope)?;
            log::info!("SCIENTIFIC_LOSSLESS_LEAF reused=true thread={:?} decoded_sha={} decoded_bytes={} encoded_bytes={} decoded_hash_ns={} object_wait_ns={} integrity_validation_ns={} total_ns={}",std::thread::current().id(),decoded,bytes.len(),envelope.encoded_bytes,decoded_hash_ns,lock_wait_ns,verify_start.elapsed().as_nanos(),leaf_start.elapsed().as_nanos());return Ok(envelope);
        }
        let bound=zstd::zstd_safe::compress_bound(bytes.len()) as u64;
        ensure!(bound<=bytes.len() as u64+65536,"lossless leaf exceeds admitted encoder overhead bound");
        let temporary=self.root.join("encoded_objects").join(format!(".{decoded}.next"));
        match fs::symlink_metadata(&temporary) {Ok(m)=>{ensure!(m.is_file()&&!m.file_type().is_symlink(),"invalid lossless object temporary");fs::remove_file(&temporary)?;},Err(e)if e.kind()==std::io::ErrorKind::NotFound=>{},Err(e)=>return Err(e.into())}
        // Bound covers worst-case encoded bytes; no assumed compression ratio.
        let reserve_start=std::time::Instant::now();self.spend_reserved_capacity(bound.checked_add(16384).context("lossless reservation overflow")?)?;let reservation_spend_ns=reserve_start.elapsed().as_nanos();
        let file=OpenOptions::new().write(true).create_new(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&temporary)?;
        let writer=HashWriter{file,hasher:Sha256::new(),count:0,limit:bound};
        let compression_start=std::time::Instant::now();
        let mut encoder=zstd::stream::write::Encoder::new(writer,1)?;
        encoder.window_log(20)?;encoder.include_checksum(true)?;encoder.include_contentsize(true)?;
        encoder.set_pledged_src_size(Some(bytes.len() as u64))?;
        encoder.write_all(bytes)?;let writer=encoder.finish()?;let compression_ns=compression_start.elapsed().as_nanos();
        let sync_start=std::time::Instant::now();writer.file.sync_all()?;let object_fsync_ns=sync_start.elapsed().as_nanos();
        let encoded=format!("{:x}",writer.hasher.finalize());let encoded_bytes=writer.count;
        drop(writer.file);
        let envelope=LosslessObjectEnvelope{schema_version:1,codec:"zstd.frame.v1".into(),encoder_version:zstd::zstd_safe::version_string().into(),window_log:20,
            encoded_sha256:encoded.clone(),encoded_bytes,decoded_sha256:decoded.clone(),decoded_bytes:bytes.len() as u64};
        let destination=self.root.join("encoded_objects").join(&encoded);
        let newly_stored=!destination.exists();
        if !newly_stored {self.verify_encoded_file(&envelope)?;fs::remove_file(&temporary)?;}
        else {fs::rename(&temporary,&destination)?;}
        sync_dir(&self.root.join("encoded_objects"))?;
        // Verify actual decoder stream and original bytes before publishing any
        // envelope reference. This is native artifact integrity, not a test.
        let verify_start=std::time::Instant::now();self.verify_lossless_object(&envelope)?;let integrity_validation_ns=verify_start.elapsed().as_nanos();
        let index_start=std::time::Instant::now();let index_bytes=serde_json::to_vec(&envelope)?;ensure!(index_bytes.len()<=16384,"lossless index exceeds reservation");
        self.publish_lossless_index(&index,&index_bytes)?;let index_publication_ns=index_start.elapsed().as_nanos();
        let actual_new=index_bytes.len() as u64+if newly_stored{encoded_bytes}else{0};
        let unused=bound.checked_add(16384).context("lossless reserve overflow")?.checked_sub(actual_new).context("lossless object exceeded reservation")?;
        let credit_start=std::time::Instant::now();self.return_reserved_capacity(unused)?;let reservation_credit_ns=credit_start.elapsed().as_nanos();
        log::info!("SCIENTIFIC_LOSSLESS_LEAF reused=false thread={:?} decoded_sha={} decoded_bytes={} encoded_bytes={} decoded_hash_ns={} object_wait_ns={} compression_ns={} object_fsync_ns={} integrity_validation_ns={} index_publication_ns={} reservation_spend_ns={} reservation_credit_ns={} total_ns={}",std::thread::current().id(),decoded,bytes.len(),encoded_bytes,decoded_hash_ns,lock_wait_ns,compression_ns,object_fsync_ns,integrity_validation_ns,index_publication_ns,reservation_spend_ns,reservation_credit_ns,leaf_start.elapsed().as_nanos());
        Ok(envelope)
    }
    pub(super) fn verify_encoded_file(&self,envelope:&LosslessObjectEnvelope)->Result<()> {
        validate_digest(&envelope.encoded_sha256)?;
        let mut file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC)
            .open(self.root.join("encoded_objects").join(&envelope.encoded_sha256))?;
        ensure!(file.metadata()?.is_file() && file.metadata()?.len()==envelope.encoded_bytes,"lossless encoded extent/type mismatch");
        let mut hash=Sha256::new();let mut buffer=[0u8;65536];let mut count=0u64;
        loop{let n=file.read(&mut buffer)?;if n==0{break;}count+=n as u64;ensure!(count<=envelope.encoded_bytes,"lossless encoded object grew");hash.update(&buffer[..n]);}
        ensure!(count==envelope.encoded_bytes && format!("{:x}",hash.finalize())==envelope.encoded_sha256,"lossless encoded hash mismatch");Ok(())
    }
    fn decode_lossless_to(&self,envelope:&LosslessObjectEnvelope,mut output:impl Write)->Result<()> {
        if envelope.codec==CHUNKED_FIELD_CODEC || envelope.codec=="prism.chronicle-spike-archive.v1" {
            let mut input=self.open_lossless_reader(envelope)?;
            let mut count=0u64;let mut hash=Sha256::new();let mut slab=[0u8;65536];
            loop {let n=input.read(&mut slab)?;if n==0{break;}
                count=count.checked_add(n as u64).context("chunked field decoded extent overflow")?;
                ensure!(count<=envelope.decoded_bytes,"chunked field exceeded exact decoded extent");
                hash.update(&slab[..n]);output.write_all(&slab[..n])?;}
            ensure!(count==envelope.decoded_bytes && format!("{:x}",hash.finalize())==envelope.decoded_sha256,
                "chunked field decoded identity differs");return Ok(());
        }
        ensure!(envelope.schema_version==1 && envelope.codec=="zstd.frame.v1" && envelope.window_log<=20,
            "unsupported lossless CAS envelope");validate_digest(&envelope.decoded_sha256)?;
        self.verify_encoded_file(envelope)?;
        let file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC)
            .open(self.root.join("encoded_objects").join(&envelope.encoded_sha256))?;
        let mut decoder=zstd::stream::read::Decoder::new(file)?;decoder.window_log_max(20)?;
        let mut hasher=Sha256::new();let mut count=0u64;let mut buffer=[0u8;65536];
        loop{let n=decoder.read(&mut buffer)?;if n==0{break;}count=count.checked_add(n as u64).context("decoded count overflow")?;
            ensure!(count<=envelope.decoded_bytes,"decoded object exceeds admitted exact extent");hasher.update(&buffer[..n]);output.write_all(&buffer[..n])?;}
        ensure!(count==envelope.decoded_bytes && format!("{:x}",hasher.finalize())==envelope.decoded_sha256,"lossless decoded size/content differs from original field");Ok(())
    }
    pub(super) fn lossless_file_stamp(&self,envelope:&LosslessObjectEnvelope)->Result<LosslessFileStamp> {
        use std::os::unix::fs::MetadataExt;
        validate_digest(&envelope.encoded_sha256)?;
        let path=self.root.join("encoded_objects").join(&envelope.encoded_sha256);
        let metadata=fs::symlink_metadata(&path)?;
        ensure!(metadata.is_file() && !metadata.file_type().is_symlink() && metadata.len()==envelope.encoded_bytes,
            "immutable encoded object inode/type/extent differs");
        Ok(LosslessFileStamp{dev:metadata.dev(),inode:metadata.ino(),bytes:metadata.len(),mtime:metadata.mtime(),
            mtime_ns:metadata.mtime_nsec(),ctime:metadata.ctime(),ctime_ns:metadata.ctime_nsec()})
    }
    pub(super) fn verify_lossless_object(&self,envelope:&LosslessObjectEnvelope)->Result<()> {
        ensure!(envelope.schema_version==1 && matches!(envelope.codec.as_str(),"zstd.frame.v1"|CHUNKED_FIELD_CODEC|"prism.chronicle-spike-archive.v1") && envelope.window_log<=20,
            "unsupported lossless CAS envelope");validate_digest(&envelope.decoded_sha256)?;
        if envelope.codec==CHUNKED_FIELD_CODEC {
            // A stable manifest inode cannot witness unchanged referenced chunk
            // inodes. Each immutable member keeps its own hash/stamp witness.
            for chunk in self.read_chunked_field_manifest(envelope)?.chunks {self.verify_lossless_object(&chunk)?;}
        }
        if envelope.codec=="prism.chronicle-spike-archive.v1" {self.verify_native_archive_members(envelope)?;}
        let stamp=self.lossless_file_stamp(envelope)?;
        if self.verified_lossless.lock().map_err(|_|anyhow::anyhow!("immutable verification witness poisoned"))?
            .get(&envelope.encoded_sha256).is_some_and(|(saved,prior)|saved==envelope && prior==&stamp) {return Ok(());}
        self.decode_lossless_to(envelope,std::io::sink())?;
        ensure!(stamp==self.lossless_file_stamp(envelope)?,"immutable encoded object changed during verification");
        let mut witnesses=self.verified_lossless.lock().map_err(|_|anyhow::anyhow!("immutable verification witness poisoned"))?;
        // Metadata memory is bounded independently of native history length.
        // Eviction costs fresh verification; it never weakens admission.
        if witnesses.len()>=32768 {witnesses.clear();}
        witnesses.insert(envelope.encoded_sha256.clone(),(envelope.clone(),stamp));Ok(())
    }
    pub(super) fn read_lossless_object(&self,envelope:&LosslessObjectEnvelope,maximum:u64)->Result<Vec<u8>> {
        ensure!(envelope.decoded_bytes<=maximum,"decoded field exceeds preallocation admission");
        let mut bytes=Vec::new();bytes.try_reserve_exact(usize::try_from(envelope.decoded_bytes)?)?;
        self.decode_lossless_to(envelope,&mut bytes)?;Ok(bytes)
    }
}

struct DecodedHashWriter<W:Write> {inner:W,hasher:Sha256,count:u64,expected:u64}
impl<W:Write> Write for DecodedHashWriter<W> {
    fn write(&mut self,bytes:&[u8])->std::io::Result<usize> {
        if self.count.checked_add(bytes.len() as u64).is_none_or(|n|n>self.expected){return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,"decoded scientific stream exceeds exact size"));}
        let n=self.inner.write(bytes)?;self.hasher.update(&bytes[..n]);self.count+=n as u64;Ok(n)
    }
    fn flush(&mut self)->std::io::Result<()>{self.inner.flush()}
}
impl CheckpointStore {
    /// Two bounded passes preserve decoded content addressing without retaining
    /// an entire host history wire image. The pause keeps the source immutable.
    pub(super) fn encode_lossless_stream(&self,decoded_bytes:u64,
        produce:&mut dyn FnMut(&mut dyn Write)->Result<()>)->Result<LosslessObjectEnvelope> {
        if decoded_bytes>SCIENTIFIC_CHUNK_BYTES as u64 {return self.encode_chunked_scientific_stream(decoded_bytes,produce);}
        let mut measure=DecodedHashWriter{inner:std::io::sink(),hasher:Sha256::new(),count:0,expected:decoded_bytes};
        produce(&mut measure)?;ensure!(measure.count==decoded_bytes,"scientific stream shorter than admitted extent");
        let decoded=format!("{:x}",measure.hasher.finalize());let _object=self.lossless_publication_guard(&decoded)?;let index=self.encoded_index_path(&decoded);
        if index.exists() {
            let envelope:LosslessObjectEnvelope=serde_json::from_slice(&read_bounded(&index,16384)?)?;
            ensure!(envelope.decoded_sha256==decoded && envelope.decoded_bytes==decoded_bytes,"stream lossless index identity mismatch");
            self.verify_lossless_object(&envelope)?;return Ok(envelope);
        }
        let bound=zstd::zstd_safe::compress_bound(usize::try_from(decoded_bytes)?) as u64;
        ensure!(bound<=decoded_bytes+65536,"streamed leaf exceeds admitted encoder overhead bound");
        self.spend_reserved_capacity(bound.checked_add(16384).context("stream compression reservation overflow")?)?;
        let temporary=self.root.join("encoded_objects").join(format!(".{decoded}.next"));
        match fs::symlink_metadata(&temporary) {Ok(m)=>{ensure!(m.is_file()&&!m.file_type().is_symlink(),"invalid stream temporary");fs::remove_file(&temporary)?;},Err(e)if e.kind()==std::io::ErrorKind::NotFound=>{},Err(e)=>return Err(e.into())}
        let writer=HashWriter{file:OpenOptions::new().write(true).create_new(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&temporary)?,hasher:Sha256::new(),count:0,limit:bound};
        let mut encoder=zstd::stream::write::Encoder::new(writer,1)?;
        encoder.window_log(20)?;encoder.include_checksum(true)?;encoder.include_contentsize(true)?;encoder.set_pledged_src_size(Some(decoded_bytes))?;
        let mut logical=DecodedHashWriter{inner:encoder,hasher:Sha256::new(),count:0,expected:decoded_bytes};
        produce(&mut logical)?;
        ensure!(logical.count==decoded_bytes && format!("{:x}",logical.hasher.finalize())==decoded,"scientific source changed between hash/encode passes");
        let writer=logical.inner.finish()?;writer.file.sync_all()?;
        let encoded=format!("{:x}",writer.hasher.finalize());let encoded_bytes=writer.count;drop(writer.file);
        let envelope=LosslessObjectEnvelope{schema_version:1,codec:"zstd.frame.v1".into(),encoder_version:zstd::zstd_safe::version_string().into(),window_log:20,
            encoded_sha256:encoded.clone(),encoded_bytes,decoded_sha256:decoded,decoded_bytes};
        let destination=self.root.join("encoded_objects").join(&encoded);let newly_stored=!destination.exists();
        if newly_stored{fs::rename(&temporary,&destination)?;}else{self.verify_encoded_file(&envelope)?;fs::remove_file(&temporary)?;}
        sync_dir(&self.root.join("encoded_objects"))?;self.verify_lossless_object(&envelope)?;
        let metadata=serde_json::to_vec(&envelope)?;ensure!(metadata.len()<=16384,"stream envelope exceeds admitted metadata");
        self.publish_lossless_index(&index,&metadata)?;
        let actual=metadata.len() as u64+if newly_stored{encoded_bytes}else{0};
        self.return_reserved_capacity(bound.checked_add(16384).context("stream compression bound overflow")?.checked_sub(actual).context("stream compression exceeded reservation")?)?;
        Ok(envelope)
    }
}

impl CheckpointStore {
    pub(super) fn publish_lossless_index(&self,index:&Path,bytes:&[u8])->Result<()> {
        ensure!(index.parent()==Some(self.root.join("encoded_objects").as_path()),"lossless index escapes store");
        if index.try_exists()? {ensure!(read_bounded(index,16384)?==bytes,"existing lossless derived index differs");return Ok(());}
        let name=index.file_name().context("lossless index locator missing")?.to_str().context("lossless index UTF8")?;
        let temporary=index.with_file_name(format!(".{name}.next"));
        match fs::symlink_metadata(&temporary){Ok(m)=>{ensure!(m.is_file()&&!m.file_type().is_symlink(),"invalid lossless metadata temporary");fs::remove_file(&temporary)?;},Err(e)if e.kind()==std::io::ErrorKind::NotFound=>{},Err(e)=>return Err(e.into())}
        super::restart_transaction::spend_branch_copy_capacity(&self.root,bytes.len() as u64)?;
        write_new(&temporary,bytes)?;fs::rename(&temporary,index)?;sync_dir(&self.root.join("encoded_objects"))?;Ok(())
    }
}


// The outer envelope commits the exact original decoded field bytes. This
// representation only divides those bytes into fixed-size immutable CAS leaves;
// it does not omit capacity, padding, inactive fields or observation histories.
const CHUNKED_FIELD_CODEC:&str="zstd.chunked-field.v1";
const SCIENTIFIC_CHUNK_BYTES:usize=4*1024*1024;
const MAX_CHUNKED_FIELD_LEAVES:usize=16384;
const MAX_CHUNKED_MANIFEST_BYTES:u64=16*1024*1024;
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct ChunkedScientificFieldV1 {schema_version:u32,chunk_bytes:u64,chunks:Vec<LosslessObjectEnvelope>}
struct ScientificChunkWriter<'a> {
    store:&'a CheckpointStore,slab:Vec<u8>,chunks:Vec<LosslessObjectEnvelope>,
    hash:Sha256,count:u64,expected:u64,
}
impl ScientificChunkWriter<'_> {
    fn seal(&mut self)->std::io::Result<()> {
        if self.slab.is_empty(){return Ok(());}
        if self.chunks.len()>=MAX_CHUNKED_FIELD_LEAVES {return Err(std::io::Error::other("scientific chunk member count exceeds admission"));}
        let envelope=self.store.encode_lossless_object(&self.slab).map_err(std::io::Error::other)?;
        self.chunks.push(envelope);self.slab.clear();Ok(())
    }
}
impl Write for ScientificChunkWriter<'_> {
    fn write(&mut self,mut bytes:&[u8])->std::io::Result<usize> {
        let length=bytes.len();
        if self.count.checked_add(length as u64).is_none_or(|n|n>self.expected){return Err(std::io::Error::other("scientific chunk producer exceeds admitted extent"));}
        self.hash.update(bytes);self.count+=length as u64;
        while !bytes.is_empty() {
            let n=bytes.len().min(SCIENTIFIC_CHUNK_BYTES-self.slab.len());
            self.slab.extend_from_slice(&bytes[..n]);bytes=&bytes[n..];
            if self.slab.len()==SCIENTIFIC_CHUNK_BYTES {self.seal()?;}
        }Ok(length)
    }
    fn flush(&mut self)->std::io::Result<()>{Ok(())}
}
struct ScientificChunkReader {
    root:PathBuf,chunks:std::vec::IntoIter<LosslessObjectEnvelope>,
    current:Option<Box<dyn Read>>,current_remaining:u64,
}
impl Read for ScientificChunkReader {
    fn read(&mut self,output:&mut[u8])->std::io::Result<usize> {
        if output.is_empty(){return Ok(0);}
        loop {
            if let Some(input)=self.current.as_mut() {
                let n=input.read(output)?;
                if n>0 {
                    self.current_remaining=self.current_remaining.checked_sub(n as u64)
                        .ok_or_else(||std::io::Error::other("decoded chunk exceeds exact extent"))?;return Ok(n);
                }
                if self.current_remaining!=0{return Err(std::io::Error::other("decoded chunk truncated"));}
                self.current=None;
            }
            let Some(chunk)=self.chunks.next() else{return Ok(0);};
            let file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC)
                .open(self.root.join("encoded_objects").join(&chunk.encoded_sha256))?;
            let mut decoder=zstd::stream::read::Decoder::new(file)?;decoder.window_log_max(20)?;
            self.current_remaining=chunk.decoded_bytes;self.current=Some(Box::new(decoder));
        }
    }
}
impl CheckpointStore {
    fn read_chunked_field_manifest(&self,envelope:&LosslessObjectEnvelope)->Result<ChunkedScientificFieldV1> {
        ensure!(envelope.schema_version==1 && envelope.codec==CHUNKED_FIELD_CODEC && envelope.window_log==20
            && envelope.encoded_bytes<=MAX_CHUNKED_MANIFEST_BYTES,"unsupported chunked field envelope");
        validate_digest(&envelope.decoded_sha256)?;self.verify_encoded_file(envelope)?;
        let bytes=read_bounded(&self.root.join("encoded_objects").join(&envelope.encoded_sha256),MAX_CHUNKED_MANIFEST_BYTES)?;
        let manifest:ChunkedScientificFieldV1=serde_json::from_slice(&bytes)?;
        ensure!(manifest.schema_version==1 && manifest.chunk_bytes==SCIENTIFIC_CHUNK_BYTES as u64
            && !manifest.chunks.is_empty() && manifest.chunks.len()<=MAX_CHUNKED_FIELD_LEAVES,"chunked field manifest population differs");
        let mut total=0u64;
        for (index,chunk) in manifest.chunks.iter().enumerate() {
            ensure!(chunk.schema_version==1 && chunk.codec=="zstd.frame.v1" && chunk.window_log==20
                && chunk.decoded_bytes>0 && chunk.decoded_bytes<=manifest.chunk_bytes
                && (index+1==manifest.chunks.len() || chunk.decoded_bytes==manifest.chunk_bytes),"chunked field member layout differs");
            total=total.checked_add(chunk.decoded_bytes).context("chunked field extent overflow")?;
        }
        ensure!(total==envelope.decoded_bytes,"chunked field member population differs from decoded extent");Ok(manifest)
    }
    pub(super) fn open_lossless_reader(&self,envelope:&LosslessObjectEnvelope)->Result<Box<dyn Read>> {
        if envelope.codec=="prism.chronicle-spike-archive.v1" {return self.open_native_archive_reader(envelope);}
        if envelope.codec==CHUNKED_FIELD_CODEC {
            let manifest=self.read_chunked_field_manifest(envelope)?;
            for chunk in &manifest.chunks {self.verify_lossless_object(chunk)?;}
            return Ok(Box::new(ScientificChunkReader{root:self.root.clone(),chunks:manifest.chunks.into_iter(),current:None,current_remaining:0}));
        }
        ensure!(envelope.schema_version==1 && envelope.codec=="zstd.frame.v1" && envelope.window_log<=20,"unknown lossless reader codec");
        self.verify_encoded_file(envelope)?;
        let file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC)
            .open(self.root.join("encoded_objects").join(&envelope.encoded_sha256))?;
        let mut decoder=zstd::stream::read::Decoder::new(file)?;decoder.window_log_max(20)?;Ok(Box::new(decoder))
    }
    fn encode_chunked_scientific_stream(&self,decoded_bytes:u64,produce:&mut dyn FnMut(&mut dyn Write)->Result<()>)->Result<LosslessObjectEnvelope> {
        ensure!(decoded_bytes>SCIENTIFIC_CHUNK_BYTES as u64
            && decoded_bytes.div_ceil(SCIENTIFIC_CHUNK_BYTES as u64)<=MAX_CHUNKED_FIELD_LEAVES as u64,"scientific field exceeds bounded chunk population");
        let mut slab=Vec::new();slab.try_reserve_exact(SCIENTIFIC_CHUNK_BYTES)?;
        let mut chunks=Vec::new();chunks.try_reserve_exact(usize::try_from(decoded_bytes.div_ceil(SCIENTIFIC_CHUNK_BYTES as u64))?)?;
        let mut writer=ScientificChunkWriter{store:self,slab,chunks,hash:Sha256::new(),count:0,expected:decoded_bytes};
        produce(&mut writer)?;writer.seal()?;
        ensure!(writer.count==decoded_bytes,"scientific chunk producer shorter than admitted field");
        ensure!(writer.chunks.iter().try_fold(0u64,|total,chunk|total.checked_add(chunk.decoded_bytes).context("chunk sum overflow"))?==decoded_bytes,
            "chunk callback lost or repeated a bounded input range");
        let decoded=format!("{:x}",writer.hash.finalize());
        let _object=self.lossless_publication_guard(&decoded)?;
        let index=self.encoded_index_path(&decoded);
        if index.exists() {
            let previous:LosslessObjectEnvelope=serde_json::from_slice(&read_bounded(&index,16384)?)?;
            ensure!(previous.decoded_sha256==decoded && previous.decoded_bytes==decoded_bytes,"decoded scientific codec index identity differs");
            self.verify_lossless_object(&previous)?;return Ok(previous);
        }
        let manifest=ChunkedScientificFieldV1{schema_version:1,chunk_bytes:SCIENTIFIC_CHUNK_BYTES as u64,chunks:writer.chunks};
        let bytes=serde_json::to_vec(&manifest)?;ensure!(bytes.len() as u64<=MAX_CHUNKED_MANIFEST_BYTES,"scientific chunk metadata exceeds bounded admission");
        let envelope=LosslessObjectEnvelope{schema_version:1,codec:CHUNKED_FIELD_CODEC.into(),encoder_version:zstd::zstd_safe::version_string().into(),window_log:20,
            encoded_sha256:hash(&bytes),encoded_bytes:bytes.len() as u64,decoded_sha256:decoded.clone(),decoded_bytes};
        let index=self.encoded_index_path(&decoded);let metadata=serde_json::to_vec(&envelope)?;
        let destination=self.root.join("encoded_objects").join(&envelope.encoded_sha256);
        let need_object=!destination.exists();let need_index=!index.exists();
        self.spend_reserved_capacity((if need_object{bytes.len() as u64}else{0})+(if need_index{metadata.len() as u64}else{0}))?;
        if need_object {write_new(&destination,&bytes)?;sync_dir(&self.root.join("encoded_objects"))?;}else{self.verify_encoded_file(&envelope)?;}
        if need_index {self.publish_lossless_index(&index,&metadata)?;}else{
            let previous:LosslessObjectEnvelope=serde_json::from_slice(&read_bounded(&index,16384)?)?;
            ensure!(previous==envelope,"same decoded field has a conflicting immutable codec index");
        }
        // Every member was hash/decoder verified at its own publication; the
        // whole decoded hash was computed from that same bounded input stream.
        let stamp=self.lossless_file_stamp(&envelope)?;
        let mut witnesses=self.verified_lossless.lock().map_err(|_|anyhow::anyhow!("immutable witness poisoned"))?;
        if witnesses.len()>=32768 {witnesses.clear();}
        witnesses.insert(envelope.encoded_sha256.clone(),(envelope.clone(),stamp));Ok(envelope)
    }
    pub(super) fn link_lossless_closure_to(&self,target:&CheckpointStore,envelope:&LosslessObjectEnvelope)->Result<()> {
        self.verify_lossless_object(envelope)?;
        if envelope.codec=="prism.chronicle-spike-archive.v1" {self.link_native_archive_closure(target,envelope)?;}
        if envelope.codec==CHUNKED_FIELD_CODEC {
            for chunk in self.read_chunked_field_manifest(envelope)?.chunks {
                self.link_lossless_closure_to(target,&chunk)?;
            }
        }
        self.link_immutable_to(target,"encoded_objects",&envelope.encoded_sha256)?;
        if envelope.codec!="prism.chronicle-spike-archive.v1" {
            let bytes=serde_json::to_vec(envelope)?;
            target.publish_lossless_index(&target.encoded_index_path(&envelope.decoded_sha256),&bytes)?;
        }Ok(())
    }
}

impl CheckpointStore {
    pub(super) fn accumulate_lossless_branch_bound(&self,envelope:&LosslessObjectEnvelope,objects:&mut std::collections::BTreeSet<(String,String)>,
        indexes:&mut std::collections::BTreeSet<String>,bound:&mut u64)->Result<()> {
        if envelope.codec=="prism.chronicle-spike-archive.v1" {self.accumulate_native_archive_branch_bound(envelope,objects,bound)?;}
        if envelope.codec==CHUNKED_FIELD_CODEC {
            for chunk in self.read_chunked_field_manifest(envelope)?.chunks {self.accumulate_lossless_branch_bound(&chunk,objects,indexes,bound)?;}
        }
        if objects.insert(("encoded_objects".into(),envelope.encoded_sha256.clone())) {
            *bound=bound.checked_add(super::restart_transaction::branch_copy_member_bound(0)?).context("encoded branch link bound overflow")?;
        }
        if indexes.insert(envelope.decoded_sha256.clone()) {
            *bound=bound.checked_add(super::restart_transaction::branch_copy_member_bound(serde_json::to_vec(envelope)?.len() as u64)?).context("encoded branch index bound overflow")?;
        }
        ensure!(objects.len()<=262144 && indexes.len()<=262144,"encoded branch metadata exceeds bounded admission");Ok(())
    }
}
