/// Calculate the Haversine distance between two points on a sphere
///
/// # Arguments
/// * `point0` - First point as [longitude, latitude] in radians
/// * `point1` - Second point as [longitude, latitude] in radians
/// * `earth_radius` - Radius of the sphere (generally expected to be 6372.8 for Earth in km)
///
/// # Returns
/// The great circle distance between the two points
///
/// # Note
/// This is not meant to be a "good" way to calculate the Haversine distance.
/// Instead, it attempts to follow, as closely as possible, the formula used in the real-world
/// question on which these homework exercises are loosely based.
pub fn reference_haversine(point0: [f64; 2], point1: [f64; 2], earth_radius: f64) -> f64 {
    let [x0, y0] = point0;
    let [x1, y1] = point1;
    
    let lat1 = y0;
    let lat2 = y1;
    let lon1 = x0;
    let lon2 = x1;

    let d_lat = lat2 - lat1;
    let d_lon = lon2 - lon1;

    let a = (d_lat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    earth_radius * c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_haversine_same_point() {
        let earth_radius = 6372.8;
        let point = [0.0, 0.0];
        let distance = reference_haversine(point, point, earth_radius);
        assert!((distance - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reference_haversine_known_distance() {
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
}
