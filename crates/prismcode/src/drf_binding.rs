//! DRF execution binding schema surface.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DrfPlanRow {
    pub row_id: String,
    pub nanl_token_u64: u64,
    pub time_state: String,
    pub replay_required: bool,
    pub payload_hash: String,
}
