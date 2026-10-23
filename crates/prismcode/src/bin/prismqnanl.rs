use clap::{Parser, Subcommand};
use prismcode::qnanl_artifact::{
    decode_file, encode_file, json_escape, packet_header, QnanlArtifactFamily, QnanlEncodeOptions,
};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(
    name = "prismqnanl",
    about = "PrismOS custom PrismCode ISA QNANL artifact encoder/decoder"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Encode one artifact into a PrismCode/QNANL packet with a receipt.
    Encode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        packet: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
        #[arg(long)]
        family: String,
        #[arg(long)]
        artifact_id: Option<String>,
    },
    /// Decode one PrismCode/QNANL packet back to canonical bytes.
    Decode {
        #[arg(long)]
        packet: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Print packet header JSON.
    Header {
        #[arg(long)]
        packet: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.command {
        Command::Encode {
            input,
            packet,
            receipt,
            family,
            artifact_id,
        } => {
            let family = QnanlArtifactFamily::from_str(&family)?;
            let artifact_id = artifact_id.unwrap_or_else(|| {
                input
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("artifact")
                    .to_string()
            });
            let receipt = encode_file(
                &input,
                &packet,
                &receipt,
                QnanlEncodeOptions::post_run_seal_only(
                    artifact_id,
                    family,
                    Some(input.clone()),
                    "prismqnanl_cli::encode",
                ),
            )?;
            println!("{}", receipt.to_json_pretty());
        }
        Command::Decode { packet, output } => {
            let decoded = decode_file(&packet, &output)?;
            let decoded_hash = blake3::hash(&decoded).to_hex().to_string();
            println!(
                concat!(
                    "{{\n",
                    "  \"status\": \"PASS\",\n",
                    "  \"packet\": \"{}\",\n",
                    "  \"output\": \"{}\",\n",
                    "  \"decoded_bytes\": {},\n",
                    "  \"decoded_hash_blake3\": \"{}\"\n",
                    "}}\n"
                ),
                json_escape(&packet.display().to_string()),
                json_escape(&output.display().to_string()),
                decoded.len(),
                decoded_hash
            );
        }
        Command::Header { packet } => {
            let header = packet_header(&packet)?;
            println!("{}", header.to_json_pretty());
        }
    }
    Ok(())
}
