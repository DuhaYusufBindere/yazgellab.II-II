def make_decision(path_probability, threshold):
    """
    Returns:
        str: Olasılık eşiğin altındaysa "Anomali (Beklenmeyen Davranış)", değilse "Normal" döner.
    """
    if path_probability < threshold:
        return "Anomali (Beklenmeyen Davranış)"
    else:
        return "Normal"
