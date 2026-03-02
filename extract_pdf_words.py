import sys
import os
import re
from collections import Counter

import fitz  # PyMuPDF

def extract_words(pdf_path, max_pages=200):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening pdf: {e}")
        return
        
    text = ""
    for i in range(min(max_pages, len(doc))):
        page = doc.load_page(i)
        text += page.get_text() + " "
            
    # Sadece harf, rakam ve tire içeren kelimeleri bulalım
    words = re.findall(r'\b[a-zA-ZçğıöşüÇĞIÖŞÜ0-9-]+\b', text)
    
    abbreviations = Counter()
    number_combos = Counter()
    regular_words = Counter()
    
    for w in words:
        if len(w) < 2:
            continue
        
        # Sadece rakamlardan oluşanları atla
        if re.match(r'^\d+$', w) or re.match(r'^[\d-]+$', w):
            continue
            
        # Kısaltmalar: Tamamen büyük harf (veya tire)
        if re.match(r'^[A-ZÇĞIÖŞÜ-]{2,}$', w):
            abbreviations[w] += 1
        # Rakam ve harf karışımı teknik değerler
        elif re.search(r'\d', w) and re.search(r'[a-zA-ZçğıöşüÇĞIÖŞÜ]', w):
            number_combos[w] += 1
        # Olağan/Uzun kelimeler (Rakam içermeyen)
        elif not re.search(r'\d', w): 
            regular_words[w.lower()] += 1
            
    # Dosyaya kaydet
    output_path = os.path.join(os.path.dirname(pdf_path), "egitim_kelimeleri.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== KISALTMALAR VE BÜYÜK HARFLİ KELİMELER (İLK 500) ===\n")
        f.write("Bunlar modelin harf harf veya tek parça olarak İngilizce/Türkçe okunuşlarını test eden kelimelerdir.\n\n")
        for w, c in abbreviations.most_common(500):
            f.write(f"{w}\n")
            
        f.write("\n\n=== RAKAM VE HARF İÇEREN TEKNİK İFADELER (İLK 500) ===\n")
        f.write("Rakam geçişleri, voltajlar, motor tipleri vb. modelin sayı ile harf arasındakı geçişini test eder.\n\n")
        for w, c in number_combos.most_common(500):
            f.write(f"{w}\n")
            
        f.write("\n\n=== NORMAL / TEKNİK İNGİLİZCE KELİMELER (İLK 1500) ===\n")
        f.write("Uzun ve farklı heceleri olan, vurgu - telaffuz test etmek için faydalı kelimeler.\n\n")
        # Sadece 4 harften uzun olanları alalım ki daha kaliteli kelimeler çıksın
        long_words = {w: c for w, c in regular_words.items() if len(w) > 4}
        for w, c in Counter(long_words).most_common(1500):
            f.write(f"{w}\n")
            
    print(f"Toplam {len(abbreviations.most_common(500)) + len(number_combos.most_common(500)) + len(Counter(long_words).most_common(1500))} kelime seçildi.")
    print(f"Kelimeler şu dosyaya kaydedildi: {output_path}")

if __name__ == "__main__":
    pdf_path = sys.argv[1]
    extract_words(pdf_path)
