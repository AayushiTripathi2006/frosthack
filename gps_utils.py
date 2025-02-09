def parse_gps_data(detections):
    """
    Parse detection data to extract GPS coordinates.

    Args:
        detections (list): Detection data, where each entry is (frame_name, latitude, longitude, detection_type)

    Returns:
        List of (latitude, longitude) coordinates
    """
    return [(det[1], det[2]) for det in detections if det[3] == "Waste Plastic"]
