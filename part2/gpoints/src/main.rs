use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;
use std::io;
use haversine::reference_haversine;

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}

fn main() {
    let mut input = String::new();

    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");

    let mut parts = input.trim().split_whitespace();
    let seed: u64 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .expect("Please type a valid seed!");
    let count: usize = parts
        .next()
        .and_then(|s| s.parse().ok())
        .expect("Please type a valid pair count!");

    println!("Generating {} pairs with seed {}...", count, seed);
    
    let pairs = generate_unit_points_pairs(count, seed);
    
    // Write points to JSON file
    let points_filename = format!("points_{}_{}.json", seed, count);
    if let Err(e) = write_json(&pairs, &points_filename) {
        eprintln!("Error writing JSON: {}", e);
        return;
    }
    println!("Successfully wrote {}", points_filename);
    
    // Compute haversine sum
    let sum = compute_haversine_sum(&pairs);
    
    // Write summary report
    let report_filename = format!("report_{}_{}.txt", seed, count);
    if let Err(e) = write_report(seed, count, sum, &report_filename) {
        eprintln!("Error writing report: {}", e);
    } else {
        println!("Successfully wrote {}", report_filename);
        println!("Sum of haversine distances: {}", sum);
    }
}

/// Generate pairs of random points on the unit sphere in spherical coordinates
fn generate_unit_points_pairs(n: usize, seed: u64) -> Vec<(Point, Point)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pairs = Vec::with_capacity(n);
    
    for _ in 0..n {
        // Generate first point
        let [x0, y0, z0] = sample_unit_sphere(&mut rng);
        let (theta0, phi0) = normalized_cartesian_to_spherical(x0, y0, z0);
        let p0 = Point::new(theta0, phi0);
        
        // Generate second point
        let [x1, y1, z1] = sample_unit_sphere(&mut rng);
        let (theta1, phi1) = normalized_cartesian_to_spherical(x1, y1, z1);
        let p1 = Point::new(theta1, phi1);
        
        pairs.push((p0, p1));
    }
    
    pairs
}

/// Sample a random point on the unit sphere using the normal distribution method
fn sample_unit_sphere(rng: &mut impl Rng) -> [f64; 3] {
    loop {
        // Sample from standard normal distribution
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);
        let z: f64 = rng.gen_range(-1.0..1.0);
        
        let length_squared = x * x + y * y + z * z;
        
        // Reject if too close to origin or outside unit sphere
        if length_squared > 0.0001 && length_squared < 1.0 {
            let length = length_squared.sqrt();
            return [x / length, y / length, z / length];
        }
    }
}

/// Convert normalized Cartesian coordinates to spherical coordinates
/// Returns (theta, phi) where theta is longitude and phi is latitude in radians
fn normalized_cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64) {
    let rho = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
    let theta = y.atan2(x);
    let phi = (z / rho).clamp(-1.0, 1.0).acos();
    (theta, phi)
}

/// Write the generated points to a JSON file using manual serialization
fn write_json(pairs: &[(Point, Point)], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut json = String::from("{\n  \"pairs\": [\n");
    
    for (i, (p0, p1)) in pairs.iter().enumerate() {
        if i > 0 {
            json.push_str(",\n");
        }
        json.push_str(&format!(
            "    {{\"x0\": {}, \"y0\": {}, \"x1\": {}, \"y1\": {}}}",
            p0.x, p0.y, p1.x, p1.y
        ));
    }
    
    json.push_str("\n  ]\n}");
    
    fs::write(filename, json)?;
    Ok(())
}

/// Compute the sum of haversine distances for all point pairs
fn compute_haversine_sum(pairs: &[(Point, Point)]) -> f64 {
    let earth_radius = 6372.8;
    pairs
        .iter()
        .map(|(p0, p1)| reference_haversine([p0.x, p0.y], [p1.x, p1.y], earth_radius))
        .sum()
}

/// Write a summary report with seed, pair count, and sum of haversine distances
fn write_report(seed: u64, count: usize, sum: f64, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let report = format!(
        "Seed: {}\nPair Count: {}\nHaversine Sum: {}\n",
        seed, count, sum
    );
    
    fs::write(filename, report)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point::new(1.5, 2.5);
        assert_eq!(p.x, 1.5);
        assert_eq!(p.y, 2.5);
    }

    #[test]
    fn test_reference_haversine() {
        // Test with known coordinates: New York to London (converted to radians)
        let ny_lon = -74.0060_f64.to_radians();
        let ny_lat = 40.7128_f64.to_radians();
        let london_lon = -0.1276_f64.to_radians();
        let london_lat = 51.5074_f64.to_radians();
        let earth_radius = 6372.8;

        let ny = [ny_lon, ny_lat];
        let london = [london_lon, london_lat];
        let distance = reference_haversine(ny, london, earth_radius);
        
        // Expected distance is approximately 5570 km
        assert!(distance > 5500.0 && distance < 5600.0, "Distance was {}", distance);
    }

    #[test]
    fn test_reference_haversine_same_point() {
        let earth_radius = 6372.8;
        let point = [0.0, 0.0];
        let distance = reference_haversine(point, point, earth_radius);
        assert!((distance - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_cartesian_to_spherical() {
        // Test conversion at (1, 0, 0) - should give theta=0, phi=π/2
        let (theta, phi) = normalized_cartesian_to_spherical(1.0, 0.0, 0.0);
        assert!((theta - 0.0).abs() < 1e-10);
        assert!((phi - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_unit_sphere() {
        let mut rng = StdRng::seed_from_u64(42);
        let [x, y, z] = sample_unit_sphere(&mut rng);
        
        // Check that the point is on the unit sphere
        let length = (x * x + y * y + z * z).sqrt();
        assert!((length - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_unit_points_pairs() {
        let pairs = generate_unit_points_pairs(10, 42);
        assert_eq!(pairs.len(), 10);
        
        // Check that all points are valid
        for (p0, p1) in pairs {
            assert!(p0.x.is_finite());
            assert!(p0.y.is_finite());
            assert!(p1.x.is_finite());
            assert!(p1.y.is_finite());
        }
    }

    #[test]
    fn test_write_json() {
        let pairs = vec![
            (Point::new(1.0, 2.0), Point::new(3.0, 4.0)),
            (Point::new(5.0, 6.0), Point::new(7.0, 8.0)),
        ];
        
        let filename = "test_points.json";
        let result = write_json(&pairs, filename);
        assert!(result.is_ok());
        
        // Clean up test file
        let _ = std::fs::remove_file(filename);
    }
}
