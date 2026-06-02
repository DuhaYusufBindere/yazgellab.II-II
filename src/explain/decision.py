def make_decision(path_probability, threshold):
    """
    Args:
        path_probability (float): Test dizisi için hesaplanmış toplam (log) olasılık.
        threshold (float): Anomali sınırını belirleyen eşik değer.
        
    Returns:
        str: Olasılık eşiğin altındaysa "Anomali (Beklenmeyen Davranış)", değilse "Normal" döner.
    """
    if path_probability < threshold:
        return "Anomali (Beklenmeyen Davranış)"
    else:
        return "Normal"


def calculate_confidence_score(path_probability, threshold, margin=1.0):
    """
    Args:
        path_probability (float): Hesaplanan yol olasılığı.
        threshold (float): Karar için kullanılan eşik değer.
        margin (float): "High" güven skoru verebilmek için gereken minimum uzaklık (fark).
        
    Returns:
        str: "High" (Yüksek) veya "Low" (Düşük) güven skoru.
    """
    difference = abs(path_probability - threshold)
    
    if difference >= margin:
        return "High"
    else:
        return "Low"
