# pulse_echo-ml

Darbe-yankı ultrasonik ölçümden çelik kalınlığı kestiren makine öğrenmesi
modelinin veri seti, öznitelik çıkarımı ve eğitim defterleri — üretilen
`.pkl` modelleri ayrı bir ölçüm uygulaması tarafından kullanılabilir.

Geliştiren: **Cem Girgin**

---

## `seshizi_ml/` paketi

DSP çözümleme (paket/yankı tespiti, çapraz korelasyon + faz eğimi
kestirimi) ve öznitelik çıkarımı burada kendi başına duran bir kopya
olarak tutuluyor — canlı bir bağlantı/bağımlılık değil, dolayısıyla bu
depo tek başına klonlanıp çalıştırılabilir. Modeli üreten kod tam olarak
burada donmuş durumda; bu, o modelin nasıl eğitildiğinin de bir kaydı.

## Kurulum

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## Dosya düzeni

```
seshizi_ml/
├── ultrasonic.py          DSP çözümleme (bkz. yukarı)
└── feature_extraction.py  Öznitelik çıkarımı

dataset/
├── index.jsonl          Her kayıt için metadata: kalınlık, ayar varyantı,
│                         pulser ayarları (kazanç/filtre/damping/PRF...),
│                         zaman/gerilim ölçeği, prob, dosya yolu
├── raw/                  595 ham osiloskop kaydı (t_s, v_volt CSV)
├── screenshots/          Her kademede bir osiloskop ekran görüntüsü
├── features_clean.csv    extract_features_dataframe() önbelleği (geçerli kayıtlar)
├── features_invalid.csv  Aynısı, kasıtlı bozuk varyant (hizli_prf_cakisma) için
└── models/
    ├── thickness_model.pkl       Baseline Gradient Boosting Regressor
    └── tuned_thickness_model.pkl GridSearchCV ile optimize edilmiş model

notebooks/
├── model_egitimi.ipynb                 Öznitelik çıkarımı → RF/GBR/KNN/MLP eğitimi,
│                                        DSP-vs-ML karşılaştırması, model kaydı
├── hiperparametre_optimizasyonu.ipynb  GridSearchCV, öznitelik seçimi (SelectFromModel),
│                                        tuned_thickness_model.pkl'i üretir
└── tum_veri_model_analizi.ipynb        Tüm 595 kayıt üzerinde (nominal + zorlu +
                                         kasıtlı bozuk) kapsamlı hata analizi,
                                         DSP-vs-ML karşılaştırma tabloları
```

## Veri nereden geliyor

Ham kayıtlar özel bir toplama betiğiyle alındı: JSR **DPR300**
pulser/receiver + Keysight **DSOX3012T** osiloskop, prob **ICHF016**,
malzeme **316 paslanmaz çelik**, kademeler **25 → 2,5 mm**.
Her kademede sabit ayarla (`GAIN=33, HP=1.0MHz, LP=5MHz, PRF=9, AMP=9,
ENERGY=HZ1, DAMPING=1`, 5 µs/böl · 2 V/böl) 20 tekrar alındı.

Buna ek olarak, modelin sabit bir kurulumun ötesinde de çalışması için
**tek değişkenli ayar varyantları** taranarak veri seti zenginleştirildi:
düşük ortalama (averaging=1), kötü kuplaj, yanlış hizalama, PRF çakışması
(kasıtlı olarak bozuk — ana regresyondan hariç tutulur, bkz. aşağı), ayrıca
HP filtre / damping / kazanç / LP filtre / pulse energy tek tek taranarak
(`index.jsonl`'deki `settings_variant` alanı). Toplam **595 kayıt**, 21
varyant grubu.

`settings_variant == "hizli_prf_cakisma"` kayıtları **kasıtlı olarak
bozuk**: PRF bilinçli olarak çok yükseğe çekilip önceki atımın yankısı
sönmeden yenisi verildi. Bunlar model eğitiminde ana veri setinden
çıkarılır; yalnızca "model bariz bozuk veriyi ayırt edebiliyor mu"
sorusunu test etmek için `features_invalid.csv`'de ayrı tutulur.

## Nasıl çalışır

1. **`model_egitimi.ipynb`** — ham CSV'lerden (`ultrasonic.analyze()` ile,
   `skip_first_packet=True` ve kalınlığa göre dinamik `max_echoes`) klasik
   DSP kestirimini ve DSP'den bağımsız istatistik/spektral öznitelikleri
   çıkarır (Hilbert zarfı, otokorelasyon, FFT), gerilimli/tabakalı bölünmeyle
   (`stratified_multi_split`) eğitim/test ayırır, birkaç regresör dener ve
   `dataset/models/thickness_model.pkl`'i kaydeder.
2. **`hiperparametre_optimizasyonu.ipynb`** — aynı öznitelik önbelleğini
   kullanıp `GridSearchCV` ile ayar taraması yapar, `SelectFromModel` ile
   öznitelik seçer, `tuned_thickness_model.pkl`'i üretir.
3. **`tum_veri_model_analizi.ipynb`** — nominal/zorlu/kasıtlı-bozuk üç
   grubu ayrı ayrı değerlendirir, kademe bazlı hata tablosu ve DSP-vs-ML
   karşılaştırması çıkarır.

Üretilen `.pkl` dosyaları `callog_pulse_echo/callog_seshizi/ml_models.py`
tarafından uygulama açılışında doğrudan yüklenir — buradaki eğitim çıktısı,
üretim uygulamasının kullandığı dosyalarla birebir aynıdır.

## Sonuçlar (özet)

Klasik DSP tek başına: MAE ≈ 8,1 mm, R² ≈ −1,74 (özellikle ince
kademelerde ve zorlu varyantlarda tutarsız). Ayarlanmış GBR modeli:
**MAE ≈ 1,25 mm, R² ≈ 0,92**; kademe bazında 0,30–0,52 mm MAE. Ayrıntılı
tablolar `tum_veri_model_analizi.ipynb` içinde.

## Yeniden üretmek

`dataset/features_clean.csv` ve `features_invalid.csv` birer önbellektir;
silinirse defterler `dataset/raw/*.csv` üzerinden yeniden hesaplar (birkaç
dakika sürer). `dataset/models/*.pkl` de aynı şekilde defterler yeniden
çalıştırılınca üretilir — elle düzenlenmemeli.
