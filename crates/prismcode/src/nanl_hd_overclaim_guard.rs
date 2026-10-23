//! NANL-HD release claim overreach guard.

pub type OverclaimResult<T> = Result<T, OverclaimError>;

#[derive(Debug, thiserror::Error)]
pub enum OverclaimError {
    #[error("PC_ERR_NANL_HD_OVERCLAIM:{0}")]
    Invalid(String),
}

pub fn validate_q4_bits_per_symbol(bits_per_symbol: f64) -> OverclaimResult<()> {
    if bits_per_symbol > 2.0 {
        return Err(OverclaimError::Invalid(
            "Q4 carries at most 2 raw bits per symbol before overhead".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_h16_not_quaternary(is_quaternary: bool) -> OverclaimResult<()> {
    if is_quaternary {
        return Err(OverclaimError::Invalid(
            "H16 is a 16-symbol alphabet, not quaternary".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_nanl_dna_claim(
    physical_claim: bool,
    physical_workflow_evidence: bool,
) -> OverclaimResult<()> {
    if physical_claim && !physical_workflow_evidence {
        return Err(OverclaimError::Invalid(
            "NANL-DNA is simulator/export only without physical synthesis/sequencing evidence"
                .to_string(),
        ));
    }
    Ok(())
}

pub fn validate_drf_lossless_claim(
    derived_state: bool,
    claims_lossless_raw: bool,
    reconstruction_proof: bool,
) -> OverclaimResult<()> {
    if derived_state && claims_lossless_raw && !reconstruction_proof {
        return Err(OverclaimError::Invalid(
            "DRF active-state reduction is not lossless raw-data compression without reconstruction proof"
                .to_string(),
        ));
    }
    Ok(())
}

pub fn validate_compression_claim(
    sample_count: u64,
    measured_ratio: Option<f64>,
    claimed_ratio: f64,
) -> OverclaimResult<()> {
    if sample_count < 10 {
        return Err(OverclaimError::Invalid(
            "performance/compression claims require minimum sample_count=10".to_string(),
        ));
    }
    let measured = measured_ratio.ok_or_else(|| {
        OverclaimError::Invalid("compression claim lacks benchmark evidence".to_string())
    })?;
    if claimed_ratio > measured {
        return Err(OverclaimError::Invalid(
            "claimed compression ratio exceeds measured benchmark evidence".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_lossy_canonical_state(
    lossy_quantized: bool,
    claims_canonical: bool,
) -> OverclaimResult<()> {
    if lossy_quantized && claims_canonical {
        return Err(OverclaimError::Invalid(
            "lossy quantized state cannot be promoted as canonical".to_string(),
        ));
    }
    Ok(())
}
