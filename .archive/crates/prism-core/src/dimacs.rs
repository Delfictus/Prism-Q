//! DIMACS graph file format parser.
//!
//! This module provides functionality to parse DIMACS .col graph files, a standard format
//! used for graph coloring benchmark problems.
//!
//! ## Format Specification
//!
//! DIMACS format follows these rules:
//! - Lines starting with 'c' are comments (ignored)
//! - Line starting with 'p edge N M' declares N vertices and M edges
//! - Lines starting with 'e U V' declare an edge between vertices U and V (1-indexed)
//!
//! ## Example
//! ```text
//! c Triangle graph example
//! p edge 3 3
//! e 1 2
//! e 2 3
//! e 1 3
//! ```
//!
//! ## Usage
//! ```no_run
//! use prism_core::dimacs::parse_dimacs_file;
//!
//! let graph = parse_dimacs_file("benchmarks/dimacs/DSJC250.5.col")?;
//! println!("Loaded graph with {} vertices and {} edges", graph.num_vertices, graph.num_edges);
//! # Ok::<(), prism_core::PrismError>(())
//! ```
//!
//! Implements PRISM GPU Plan (§2.1: Graph Representation).

use crate::{Graph, PrismError};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Helper to create a validation error for DIMACS parsing
fn parse_error(message: impl Into<String>) -> PrismError {
    PrismError::validation(message.into())
}

/// Parses a DIMACS .col file into a Graph.
///
/// This function reads a DIMACS format graph file and constructs a Graph with
/// undirected edges. Vertices are converted from 1-indexed (DIMACS standard)
/// to 0-indexed (Rust convention).
///
/// ## Arguments
/// * `path` - Path to the DIMACS .col file
///
/// ## Returns
/// * `Ok(Graph)` - Successfully parsed graph
/// * `Err(PrismError)` - I/O error or parsing error
///
/// ## Errors
/// - `PrismError::Io`: File not found or read error
/// - `PrismError::Parse`: Invalid DIMACS format (malformed problem line, invalid edge, etc.)
///
/// ## Example
/// ```no_run
/// # use prism_core::dimacs::parse_dimacs_file;
/// let graph = parse_dimacs_file("benchmarks/dimacs/myciel3.col")?;
/// assert!(graph.num_vertices > 0);
/// # Ok::<(), prism_core::PrismError>(())
/// ```
pub fn parse_dimacs_file<P: AsRef<Path>>(path: P) -> Result<Graph, PrismError> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref).map_err(|e| {
        PrismError::Internal(format!(
            "Failed to open DIMACS file '{}': {}",
            path_ref.display(),
            e
        ))
    })?;

    let reader = BufReader::new(file);

    let mut num_vertices = 0;
    let mut num_edges_declared = 0;
    let mut edges = Vec::new();
    let mut problem_line_found = false;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| {
            PrismError::Internal(format!(
                "Failed to read line {} from DIMACS file: {}",
                line_num + 1,
                e
            ))
        })?;

        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('c') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "p" => {
                // Problem line: p edge N M
                if parts.len() < 4 {
                    return Err(parse_error(format!(
                        "Invalid problem line format at line {}: expected 'p edge N M', got '{}'",
                        line_num + 1,
                        line
                    )));
                }
                if parts[1] != "edge" {
                    return Err(parse_error(format!(
                        "Unsupported problem type '{}' at line {}: only 'edge' is supported",
                        parts[1],
                        line_num + 1
                    )));
                }

                num_vertices = parts[2].parse::<usize>().map_err(|_| {
                    parse_error(format!(
                        "Invalid vertex count '{}' at line {}: must be a positive integer",
                        parts[2],
                        line_num + 1
                    ))
                })?;

                num_edges_declared = parts[3].parse::<usize>().map_err(|_| {
                    parse_error(format!(
                        "Invalid edge count '{}' at line {}: must be a positive integer",
                        parts[3],
                        line_num + 1
                    ))
                })?;

                problem_line_found = true;
            }
            "e" => {
                // Edge line: e U V
                // Must have problem line before edges
                if !problem_line_found {
                    return Err(parse_error(format!(
                        "Edge definition at line {} before problem line (expected 'p edge N M' first)",
                        line_num + 1
                    )));
                }

                if parts.len() < 3 {
                    return Err(parse_error(format!(
                        "Invalid edge line format at line {}: expected 'e U V', got '{}'",
                        line_num + 1,
                        line
                    )));
                }

                let u = parts[1].parse::<usize>().map_err(|_| {
                    parse_error(format!(
                        "Invalid vertex ID '{}' at line {}: must be a positive integer",
                        parts[1],
                        line_num + 1
                    ))
                })?;

                let v = parts[2].parse::<usize>().map_err(|_| {
                    parse_error(format!(
                        "Invalid vertex ID '{}' at line {}: must be a positive integer",
                        parts[2],
                        line_num + 1
                    ))
                })?;

                // Validate vertex IDs are in range (DIMACS uses 1-indexed)
                if u == 0 || u > num_vertices {
                    return Err(parse_error(format!(
                        "Vertex ID {} at line {} out of range [1, {}]",
                        u,
                        line_num + 1,
                        num_vertices
                    )));
                }
                if v == 0 || v > num_vertices {
                    return Err(parse_error(format!(
                        "Vertex ID {} at line {} out of range [1, {}]",
                        v,
                        line_num + 1,
                        num_vertices
                    )));
                }

                // Convert to 0-indexed and store edge
                edges.push((u - 1, v - 1));
            }
            _ => {
                // Ignore unknown line types (forward compatibility)
                log::debug!(
                    "Ignoring unknown DIMACS line type '{}' at line {}",
                    parts[0],
                    line_num + 1
                );
            }
        }
    }

    // Validate that we found a problem line
    if !problem_line_found {
        return Err(parse_error(
            "No problem line found in DIMACS file (expected 'p edge N M')",
        ));
    }

    if num_vertices == 0 {
        return Err(parse_error("Problem line declares 0 vertices"));
    }

    // Build adjacency list
    let mut adjacency = vec![Vec::new(); num_vertices];
    for (u, v) in &edges {
        // Skip self-loops
        if u == v {
            log::warn!("Skipping self-loop edge ({}, {})", u, v);
            continue;
        }
        adjacency[*u].push(*v);
        adjacency[*v].push(*u); // Undirected graph
    }

    // Deduplicate neighbors (handle duplicate edges in input file)
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    // Count actual edges (sum of neighbor counts / 2 for undirected)
    let actual_edges: usize = adjacency.iter().map(|n| n.len()).sum::<usize>() / 2;

    // Log warning if declared edge count doesn't match actual
    if actual_edges != num_edges_declared {
        log::warn!(
            "DIMACS file declared {} edges but actual edge count is {} (after deduplication)",
            num_edges_declared,
            actual_edges
        );
    }

    Ok(Graph {
        num_vertices,
        num_edges: actual_edges,
        adjacency,
        degrees: None,      // Will be computed lazily via Graph::degrees()
        edge_weights: None, // DIMACS .col files don't specify edge weights
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Creates a temporary DIMACS file with the given content
    fn create_temp_dimacs(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        file.write_all(content.as_bytes())
            .expect("Failed to write to temp file");
        file.flush().expect("Failed to flush temp file");
        file
    }

    #[test]
    fn test_parse_simple_triangle() {
        let content = "\
c Triangle graph
p edge 3 3
e 1 2
e 2 3
e 1 3
";
        let file = create_temp_dimacs(content);
        let graph = parse_dimacs_file(file.path()).unwrap();

        assert_eq!(graph.num_vertices, 3);
        assert_eq!(graph.num_edges, 3);

        // Verify adjacency structure (0-indexed)
        assert_eq!(graph.adjacency[0].len(), 2); // Vertex 0 connected to 1, 2
        assert_eq!(graph.adjacency[1].len(), 2); // Vertex 1 connected to 0, 2
        assert_eq!(graph.adjacency[2].len(), 2); // Vertex 2 connected to 0, 1

        assert!(graph.adjacency[0].contains(&1));
        assert!(graph.adjacency[0].contains(&2));
    }

    #[test]
    fn test_parse_with_comments() {
        let content = "\
c This is a comment
c Another comment
p edge 2 1
c Comment between problem and edges
e 1 2
c Trailing comment
";
        let file = create_temp_dimacs(content);
        let graph = parse_dimacs_file(file.path()).unwrap();

        assert_eq!(graph.num_vertices, 2);
        assert_eq!(graph.num_edges, 1);
        assert_eq!(graph.adjacency[0], vec![1]);
        assert_eq!(graph.adjacency[1], vec![0]);
    }

    #[test]
    fn test_parse_duplicate_edges() {
        let content = "\
p edge 2 3
e 1 2
e 1 2
e 2 1
";
        let file = create_temp_dimacs(content);
        let graph = parse_dimacs_file(file.path()).unwrap();

        // Should deduplicate to 1 edge
        assert_eq!(graph.num_vertices, 2);
        assert_eq!(graph.num_edges, 1);
        assert_eq!(graph.adjacency[0], vec![1]);
        assert_eq!(graph.adjacency[1], vec![0]);
    }

    #[test]
    fn test_parse_isolated_vertices() {
        let content = "\
p edge 5 2
e 1 2
e 3 4
";
        let file = create_temp_dimacs(content);
        let graph = parse_dimacs_file(file.path()).unwrap();

        assert_eq!(graph.num_vertices, 5);
        assert_eq!(graph.num_edges, 2);

        // Vertex 4 (0-indexed) should be isolated
        assert_eq!(graph.adjacency[4].len(), 0);
    }

    #[test]
    fn test_parse_self_loop_ignored() {
        let content = "\
p edge 3 3
e 1 2
e 1 1
e 2 3
";
        let file = create_temp_dimacs(content);
        let graph = parse_dimacs_file(file.path()).unwrap();

        // Self-loop should be ignored
        assert_eq!(graph.num_edges, 2);
    }

    #[test]
    fn test_parse_error_no_problem_line() {
        let content = "\
e 1 2
e 2 3
";
        let file = create_temp_dimacs(content);
        let result = parse_dimacs_file(file.path());

        assert!(result.is_err());
        match result {
            Err(PrismError::ValidationError(message)) => {
                assert!(
                    message.contains("before problem line"),
                    "Expected 'before problem line' error, got: {}",
                    message
                );
            }
            _ => panic!("Expected ValidationError, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_error_invalid_problem_line() {
        let content = "p edge 3\n";
        let file = create_temp_dimacs(content);
        let result = parse_dimacs_file(file.path());

        assert!(result.is_err());
        match result {
            Err(PrismError::ValidationError(message)) => {
                assert!(message.contains("Invalid problem line format"));
            }
            _ => panic!("Expected ValidationError, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_error_invalid_edge_format() {
        let content = "\
p edge 3 1
e 1
";
        let file = create_temp_dimacs(content);
        let result = parse_dimacs_file(file.path());

        assert!(result.is_err());
        match result {
            Err(PrismError::ValidationError(message)) => {
                assert!(message.contains("Invalid edge line format"));
            }
            _ => panic!("Expected ValidationError, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_error_vertex_out_of_range() {
        let content = "\
p edge 3 1
e 1 5
";
        let file = create_temp_dimacs(content);
        let result = parse_dimacs_file(file.path());

        assert!(result.is_err());
        match result {
            Err(PrismError::ValidationError(message)) => {
                assert!(message.contains("out of range"));
            }
            _ => panic!("Expected ValidationError, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_error_invalid_vertex_id() {
        let content = "\
p edge 3 1
e 1 abc
";
        let file = create_temp_dimacs(content);
        let result = parse_dimacs_file(file.path());

        assert!(result.is_err());
        match result {
            Err(PrismError::ValidationError(message)) => {
                assert!(message.contains("Invalid vertex ID"));
            }
            _ => panic!("Expected ValidationError, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_error_zero_vertex_id() {
        let content = "\
p edge 3 1
e 0 1
";
        let file = create_temp_dimacs(content);
        let result = parse_dimacs_file(file.path());

        assert!(result.is_err());
        match result {
            Err(PrismError::ValidationError(message)) => {
                assert!(message.contains("out of range"));
            }
            _ => panic!("Expected ValidationError, got {:?}", result),
        }
    }

    #[test]
    fn test_parse_empty_graph() {
        let content = "p edge 5 0\n";
        let file = create_temp_dimacs(content);
        let graph = parse_dimacs_file(file.path()).unwrap();

        assert_eq!(graph.num_vertices, 5);
        assert_eq!(graph.num_edges, 0);
        assert_eq!(graph.adjacency.len(), 5);
        for adj in &graph.adjacency {
            assert!(adj.is_empty());
        }
    }

    #[test]
    fn test_parse_nonexistent_file() {
        let result = parse_dimacs_file("/nonexistent/path/to/file.col");
        assert!(result.is_err());
        match result {
            Err(PrismError::Internal(message)) => {
                assert!(message.contains("Failed to open DIMACS file"));
            }
            _ => panic!("Expected Internal error, got {:?}", result),
        }
    }
}
