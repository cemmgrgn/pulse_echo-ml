"""Ultrasonic pulse-echo sound velocity analysis.

Derives the speed of sound in a block from the echo train the oscilloscope
captured. A pulse is launched into the block; it reflects off the back wall,
returns, reflects off the front face, and travels again. Each round trip
takes ``2d/c``, so successive back-wall echoes are evenly spaced in time and

    c = 2d / dt

Measurement rule
----------------
An echo is a **packet** -- several cycles of RF -- not a single spike. The
timing feature is therefore not "the peak of the echo" but a *specific cycle*
of it, and the same cycle must be picked in every packet. This module picks
the Nth local maximum (default: the 2nd) and, independently, the Nth local
minimum, producing two families of estimates whose disagreement is a direct
measure of the error introduced by the feature choice itself.

Why several echoes and not just the first pair
-----------------------------------------------
The largest packet on screen is usually the **excitation pulse**, not a
back-wall echo. Its shape differs from an echo's, so matching "2nd peak of
the excitation" against "2nd peak of an echo" carries a systematic offset.
Consecutive echo-to-echo pairs (2->3, 3->4) are immune to it, because both
ends of the interval are the same kind of event. That, more than the extra
averaging, is why the 3rd and 4th reflections matter.

Sub-sample timing
-----------------
Feature times are refined by fitting a parabola through the three samples
around each extremum. Without this the timing is pinned to the sample grid
and the achievable resolution is the sample interval; with it, the resolution
is set by noise instead. This is where the sub-nanosecond timing comes from,
not from the instrument's sample rate.

Limitation
----------
The estimates within one frame share timestamps (the 1->2 and 1->3 intervals
both use the first packet), so they are **not statistically independent**.
Their spread tells you whether the picking is consistent; it is not a
rigorous type-A uncertainty. A real type-A term comes from repeating the
measurement -- see `uncertainty()`.
"""

import numpy as np

#: A packet is a region whose envelope rises this many times above the quiet
#: floor. The threshold is set from the **noise**, not from the largest
#: packet: the excitation pulse is several times taller than the first echo,
#: so a fraction-of-maximum threshold silently drops the 3rd and 4th echoes
#: -- exactly the ones the averaging depends on.
#:
#: The factor is small because the floor it multiplies is already high: the
#: envelope takes a moving maximum, so the quiet region sits at roughly
#: three standard deviations of the noise rather than at zero. A larger
#: factor pushed the threshold above the first echo on a noisy record and
#: the whole frame was reported as empty instead of degrading to two echoes.
PACKET_NOISE_FACTOR = 3.0

#: Floor under the noise-based threshold, as a fraction of the largest
#: envelope value. Guards the opposite case: a record with almost no noise,
#: where five times the floor would still be inside the quantisation dust.
PACKET_MIN_RATIO = 0.02

#: A local extremum only counts as a feature point if it reaches this
#: fraction of its own packet's envelope peak. Without it the small ripples
#: on a packet's rising flank get counted as cycles and the "2nd peak" lands
#: on noise -- the same failure `defib.MIN_PHASE_SAMPLES` guards against.
FEATURE_PROMINENCE_RATIO = 0.20

#: Packet duration divided by echo spacing, above which the reflections can
#: no longer be told apart.
#:
#: Two packets of duration T whose centres are `spacing` apart just touch
#: when T equals the spacing, so they only stay separated -- with a quiet
#: stretch between them for the envelope to fall back to the floor -- while
#: T stays under half of it. Past that, neighbouring packets start sharing
#: cycles and the later echoes disappear into the interference instead of
#: being detected.
#:
#: This is a property of the probe and the block: no setting in the
#: application can recover from it, so it is reported rather than worked
#: around.
OVERLAP_RATIO = 0.5

#: Phase agreement below which the feature picking is not to be trusted.
#: 1.0 means every packet was timed on the same cycle; 0 means the picks are
#: spread uniformly around the cycle, so the velocity is arithmetic on noise.
COHERENCE_WARN = 0.90

#: Bir gidiş-dönüşe düşmesi gereken en az örnek sayısı.
#:
#: Zamanlama, paket içindeki çevrimleri ayırt edebilmeye dayanıyor; bir
#: gidiş-dönüşe birkaç düzine örnek düştüğünde paketin kendisi birkaç
#: örnekten ibaret kalıyor ve çevrim diye bir şey görünmüyor. Daha da
#: kötüsü, taşıyıcı Nyquist'in üstündeyse **katlanıyor**: ölçülen frekans
#: makul bir sayı olarak çıkıyor, örnek/çevrim oranı bol görünüyor ve
#: hiçbir şey yanlış gibi durmuyor. Bu yüzden denetim ölçülen frekansa
#: değil, kalınlıktan bilinen gidiş-dönüşe bakıyor.
MIN_SAMPLES_PER_ROUND_TRIP = 300

#: Kayıt penceresine sığabilecek en fazla gidiş-dönüş, istenen yankı
#: sayısının üstüne. Bundan fazlası zaman tabanının gereğinden yavaş
#: olduğunu gösterir: yankılar ekranda üst üste yığılır ve her birine
#: düşen örnek sayısı gereksiz yere azalır.
MAX_ROUND_TRIPS_IN_RECORD = 3

#: Bir örneğin "sınıra dayanmış" sayılması için kaydın uçlarına yakınlığı,
#: tam aralığın oranı olarak; ve ölçüme giren bir pakette bu kadar örnek
#: sınırdaysa uyarı verilir.
#:
#: Doymuş alıcı, çapraz denetimin göremediği tek başarısızlık: zarf da
#: çevrim zamanlaması da **aynı** bozulmadan geçtiği için ikisi birbirini
#: tutar ve sonuç tutarlı görünürken birkaç yüzde yanlış olur. Sebebi
#: dalganın kendisinde aramaktan başka yol yok.
#:
#: Ana darbenin kırpılması normal ve beklenen; bu yüzden denetim yalnızca
#: ölçüme giren paketlere bakıyor.
CLIP_RAIL_TOLERANCE = 0.005
CLIP_WARN_FRACTION = 0.02

#: Half-width of the digital band-pass, as a fraction of the probe's centre
#: frequency. 0.6 keeps roughly 0.4*f0 to 1.6*f0 -- wider than the probe's
#: own bandwidth, so the echo shape is left alone, but far away from both
#: the baseline wander below it and the quantisation grass above it.
BANDPASS_WIDTH_RATIO = 0.6

#: Regions closer together than this fraction of the round-trip time belong
#: to the same echo and are merged.
#:
#: A real probe does not ring down smoothly. The envelope of one echo dips
#: and comes back -- on measured data a 25 mm block gave a 2.1 us packet
#: followed by a 0.57 us gap and then another 1.1 us of ring-down. Merging
#: only across a carrier period (the first thing tried) left that tail
#: standing as its own packet, and the "consecutive pair" 1->2 then measured
#: the distance from an echo to **its own tail** rather than to the next
#: echo. The velocity came out three times too high and, because the tail is
#: a stable feature, it came out that way every single frame.
#:
#: Anything closer than a third of a round trip cannot be a separate
#: reflection, so merging at 0.4 is safe against joining two real echoes
#: while still swallowing the ring-down.
PACKET_MERGE_RATIO = 0.4

#: An echo plus its ring-down cannot outlast the gap to the next echo --
#: past that the two would overlap and there would be nothing to measure.
#: The cap stops merging from chaining: each merge moves the region's end
#: forward, which brings the next hump within reach, and on measured data
#: that walked the last packet all the way to the end of the record. Its
#: centroid then moved with it and the cycle anchoring drifted.
MAX_PACKET_RATIO = 0.7

#: Bir paketin yankı ızgarasına oturması için, gidiş-dönüşün tam katından
#: sapması bunu geçmemeli. 0.25: yarım yankı aralığının yarısı — gerçek bir
#: yansımanın bu kadar kayması mümkün değil, çınlama kuyruğu ise ızgaranın
#: çok uzağına düşer.
GRID_TOLERANCE = 0.25

#: Autocorrelation height below which the envelope shows no usable
#: periodicity -- a single echo, or noise.
MIN_AUTOCORRELATION = 0.12

#: How far the final answer may sit from the coarse round trip measured
#: from the envelope before it is called into question. The two are found
#: completely differently -- one from the periodicity of the whole train,
#: one from individual cycle timings -- so a disagreement means the cycle
#: picking landed somewhere it should not have.
ROUND_TRIP_TOLERANCE = 0.25

#: Bu farkın üstünde sonuç **bildirilmiyor**, yalnızca uyarılmıyor.
#: Kaba ve ince sayı bu kadar ayrıştığında ikisinden en az biri tamamen
#: yanlıştır ve hangisi olduğu bilinemez. Ekranda büyük puntoyla duran bir
#: sayının altına uyarı yazmak yetmez -- deftere geçen sayının kendisidir.
ROUND_TRIP_REJECT = 0.5

#: Envelope level, as a fraction of a packet's **own** peak, that bounds the
#: region the anchor is computed over.
#:
#: Detection and timing need thresholds of opposite kinds, and using one for
#: both is a bug. Detection has to be absolute -- referred to the noise --
#: or a weak fourth echo is never found at all. Timing has to be relative:
#: with a spike-excited probe the packet is asymmetric, a sharp onset
#: followed by a long ring-down, so a strong echo keeps its tail above any
#: fixed level far longer than a weak one does. Cut on an absolute level and
#: the first echo's region is a different *shape* from the fourth's, their
#: centroids sit at different offsets from their onsets, and the cycle index
#: slips between packets -- which is exactly the failure the anchor exists
#: to prevent.
#:
#: Half of the packet's own peak is the usual leading-edge convention and
#: makes the region geometrically similar in every packet, whatever its
#: amplitude.
ANCHOR_THRESHOLD_RATIO = 0.5

#: Minimum spacing between two accepted feature points, in carrier periods.
#: Below one period they would be the same cycle; 0.6 leaves room for the
#: period estimate to be a little off without ever merging two real cycles.
MIN_FEATURE_SEPARATION_PERIODS = 0.6

#: A real packet lasts at least this many carrier periods. Expressed in
#: periods rather than samples because the sample count that corresponds to
#: "too short to be an echo" changes with the timebase; a fixed sample count
#: let noise spikes through at fast sweeps and rejected real echoes at slow
#: ones.
MIN_PACKET_PERIODS = 0.5

#: Half-width, in carrier periods, of the window that cycles are counted in.
#:
#: Cycles are indexed inside a fixed-width window anchored on the packet's
#: envelope **centroid**, not on where the packet crosses the threshold and
#: not on where its envelope peaks.
#:
#: A threshold crossing sits closer to the centre on a weak echo than on a
#: strong one, so counting from there makes "the 2nd peak" mean a different
#: cycle in each packet. The envelope maximum is amplitude-independent but
#: jitters by a good fraction of a period under noise and quantisation, and
#: when it moves past a cycle boundary the window gains or loses a peak at
#: its edge -- the index then slips by a whole cycle and the interval comes
#: out one period short or long. That failure is stable and plausible-
#: looking, which makes it the dangerous one.
#:
#: The centroid averages the whole packet instead of trusting one sample, so
#: it holds still to well under a period and the window keeps the same set
#: of cycles in every packet.
FEATURE_WINDOW_PERIODS = 2.5

#: Reference longitudinal velocities (m/s), shown next to the measurement so
#: a gross error (wrong thickness, wrong packet) is obvious at a glance.
REFERENCE_VELOCITY = {
    # 316 paslanmaz ayrı duruyor: karbon çelikten belirgin biçimde yavaş ve
    # laboratuvardaki basamaklı blok bu malzemeden. 25 mm basamakta ölçülen
    # 5744 m/s bu değeri doğruladı.
    "paslanmaz_316": 5740.0,
    # Doku eşdeğeri fantom: tıbbi ultrasonda kabul edilen değer. Çelikten
    # dört kat yavaş olduğu için aynı kalınlıkta yankı aralığı dört kat
    # uzun; zaman tabanı ve nokta sayısı buna göre kendiliğinden ayarlanıyor.
    "fantom": 1540.0,
    "su": 1480.0,
    "aluminyum": 6320.0,
    "celik": 5920.0,
    "pirinc": 4700.0,
}

#: Feature families produced per packet.
#: Gecikme kestirim yöntemleri.
#:
#: Tek tek tepe/çukur noktalarını zamanlamak yerine yankının **tamamı**
#: kullanılıyor. Nedeni deneyle görüldü: hangi çevrimin seçileceği ayrık bir
#: karar ve belirginlik eşiğinin sınırındaki bir çevrim paketten pakete
#: girip çıkıyor; indeks bir kayınca ölçülen süre tam bir taşıyıcı periyot
#: şaşıyor ve sonuç kararlı, makul, tamamen yanlış çıkıyor. Dalganın
#: bütününü kullanan yöntemlerde böyle bir seçim yok.
#:
#: İkisi de yerleşik yöntemler ve **bağımsız** çalışıyor; ayrışmaları tek
#: başına bir sonucun güvenilmezliğini gösterir:
#:
#: * ``xcorr`` — iki ekonun çapraz korelasyonu. ASTM C1331'in tarif ettiği
#:   yöntem (ileri seramiklerde ultrasonik hız ölçümü). Korelasyonun zarfı
#:   kaba gecikmeyi çevrim belirsizliği olmadan verir, tepedeki faz ise
#:   çevrim altı çözünürlüğü.
#: * ``phase`` — çapraz tayfın faz eğimi. Faz, gecikmeyle doğrusal
#:   (φ = −ωτ); prob bant genişliği boyunca eğim uydurulur. Zamandaki tek
#:   bir noktaya değil, bandın tamamına dayanır.
METHOD_XCORR = "xcorr"
METHOD_PHASE = "phase"
METHODS = (METHOD_XCORR, METHOD_PHASE)

METHOD_LABELS = {METHOD_XCORR: "çapraz korelasyon", METHOD_PHASE: "faz eğimi"}

#: Bir ekoyu çevresinden koparan pencerenin yarı genişliği, gidiş-dönüşün
#: oranı olarak. Komşu yankıyı içeri almayacak kadar dar, paketin kuyruğunu
#: dışarıda bırakmayacak kadar geniş olmalı; yarım gidiş-dönüş ikisini de
#: sağlıyor.
GATE_HALF_WIDTH_RATIO = 0.5

#: İki yöntemin bir çift üzerinde uyuşması gereken en büyük fark, taşıyıcı
#: periyodun oranı olarak.
#:
#: Çapraz korelasyon ile faz eğimi aynı gecikmeyi bambaşka yollardan
#: buluyor; ortak bir hataları yok. Malzemedeki frekansa bağlı soğurma ve
#: pencereleme asimetrisi gibi fiziksel etkiler göz önüne alınarak tolerans
#: 0.45 periyot olarak belirlenmiştir.
PAIR_AGREEMENT_RATIO = 0.45


def analyze(times, values, thickness_m, max_echoes=4,
            skip_first_packet=False, reference_velocity=None):
    """Analyzes an echo train.

    times: seconds, values: volts, thickness_m: block thickness in metres.
    skip_first_packet: drop the leading packet because it is the excitation
    pulse rather than a back-wall echo.

    Return: dict; {"found": False, "reason": ...} when there is no usable
    echo train.
    """
    n = min(len(times), len(values))
    if n < 16:
        return {"found": False, "reason": "Yeterli veri yok"}
    if not thickness_m or float(thickness_m) <= 0:
        return {"found": False, "reason": "Blok kalınlığı girilmedi"}

    times = np.asarray(times, dtype=np.float64)[:n]
    values = np.asarray(values, dtype=np.float64)[:n]
    thickness_m = float(thickness_m)

    # Median of the **whole** record, not of a pre-trigger region: in a
    # pulse-echo capture most of the record is quiet between echoes, and
    # there is no guaranteed pre-trigger window to fall back on. RF is
    # bipolar about zero, and a DC offset shifts the envelope and therefore
    # the gate positions.
    baseline = float(np.median(values))
    values = values - baseline

    dt = _sample_interval(times)
    peak_abs = float(np.max(np.abs(values)))
    if peak_abs <= 0:
        return {"found": False, "reason": "Sinyal yok"}

    # Yakalama ayarları kalınlıkla tutarlı mı — çözümlemeye girmeden önce.
    # Ayar yanlışsa buradan sonraki her adım anlamsız sayı üretir ve hata
    # "yankı bulunamadı" gibi görünür; oysa yapılması gereken şey cihazda.
    capture = _capture_problem(times, dt, thickness_m, reference_velocity,
                               max_echoes)
    if capture:
        return {"found": False, "reason": capture}

    # Taşıyıcı frekans ham sinyalden ölçülüyor, süzülmüşten değil: bant
    # geçirenin merkezi zaten bu ölçüme dayanıyor, süzülmüş sinyalden
    # yeniden ölçmek kendi bandını doğrulamaktan başka bir şey söylemezdi.
    # Kırpma denetimi süzmeden **önceki** sinyale bakmak zorunda: bant
    # geçiren düz platoyu yuvarlar ve doymanın izini siler.
    raw_values = values
    period_samples = _carrier_period_samples(values)
    if dt:
        values = _bandpass(values, dt, 1.0 / (period_samples * dt))
        peak_abs = float(np.max(np.abs(values)))
        if peak_abs <= 0:
            return {"found": False, "reason": "Süzmeden sonra sinyal kalmadı"}
    envelope = _envelope(values, period_samples)

    # The round trip is measured from the envelope **before** packets are
    # cut, because it decides how far apart two humps must be to count as
    # separate echoes rather than one echo's ring-down.
    #
    # When the caller says the first packet is the excitation pulse rather
    # than an echo, it is also -- by a wide margin -- the loudest thing in
    # the record: the receiver is driven straight from the pulser's spike,
    # not from a reflection. Left in, it dominates the envelope's self-
    # correlation and the lag that wins is bang-to-echo, not echo-to-echo.
    # On a thin, fast-echoing block that measured round trip came out ~25%
    # too long, and every real echo after the first was then graded against
    # that wrong grid and thrown out as "doesn't fit" -- exactly the
    # "İkinci/üçüncü yankı bulunamıyor" reports on short round trips.
    # Masking the excitation out of a throwaway copy before this one
    # measurement fixes that; the packet boundaries below still see the
    # unmasked envelope; only the round-trip estimate is affected.
    round_trip_envelope = envelope
    if skip_first_packet:
        bang_regions = _packets(envelope, period_samples)
        if bang_regions:
            a, b = bang_regions[0]
            round_trip_envelope = envelope.copy()
            round_trip_envelope[a:b + 1] = float(np.median(envelope))
    coarse = _envelope_round_trip(round_trip_envelope, dt, period_samples)
    merge_samples = max_samples = None
    if coarse and dt:
        merge_samples = PACKET_MERGE_RATIO * coarse[0] / dt
        max_samples = MAX_PACKET_RATIO * coarse[0] / dt
    regions = _packets(envelope, period_samples, merge_samples, max_samples)

    warnings = []
    # Kept for diagnosis: once the excitation pulse is dropped, a frame where
    # every echo merged into one region looks identical to a frame with no
    # signal at all, and the two need very different answers.
    all_regions = list(regions)
    if skip_first_packet and regions:
        regions = regions[1:]
    if len(regions) > max_echoes:
        regions = regions[:max_echoes]
    if len(regions) < 2:
        return {"found": False,
                "reason": _too_few_packets_reason(all_regions, times,
                                                  thickness_m,
                                                  reference_velocity,
                                                  coarse[0] if coarse else None)}
    if len(regions) < max_echoes:
        warnings.append(
            "Yalnızca %d paket bulundu — istenen %d yansımanın tamamı "
            "ortalamaya katılamadı. Donanım ortalamasını artırmayı ya da "
            "dikey ölçeği küçültmeyi deneyin." % (len(regions), max_echoes))

    packets = [_packet_info(times, envelope, a, b, i, period_samples)
               for i, (a, b) in enumerate(regions, start=1)]

    # Gerçek bir yansıma, ilk yankıdan gidiş-dönüşün tam katı kadar sonra
    # gelir. Bu ızgaraya oturmayan paket yankı değildir: ölçülen veride
    # birinci yankının çınlama kuyruğu ayrı bir paket olarak görünüp
    # "ikinci yankı" sanılmış ve hız beş kat yüksek çıkmıştı. Kayıt sonunda
    # yarısı kesilmiş bir yankı da buradan eleniyor — ağırlık merkezi
    # kırpılma yüzünden kayar ve zamanlaması güvenilmezdir.
    if coarse and len(packets) > 1:
        packets, dropped = _keep_on_grid(packets, coarse[0])
        if dropped:
            warnings.append(
                "%d paket yankı ızgarasına oturmadığı için ölçüme "
                "katılmadı (çınlama kuyruğu ya da ekranın kenarında "
                "kesilmiş yankı olabilir)." % dropped)
        if len(packets) < 2:
            return {"found": False,
                    "reason": "Yankı ızgarasına oturan en az iki paket yok. "
                              "Ekranda ilk paketin arkasından 3–4 yankı "
                              "görünecek şekilde zaman tabanını ayarlayın."}
        for index, packet in enumerate(packets, start=1):
            packet["index"] = index

    noise = _noise_level(values, regions)
    for p in packets:
        p["snr_db"] = (20.0 * np.log10(p["envelope_peak"] / noise)
                       if noise > 0 and p["envelope_peak"] > 0 else None)

    carrier_s = (period_samples * dt) if dt else 0.0
    carrier_hz = (1.0 / carrier_s) if carrier_s else None

    # Her paketin ızgaradaki sırası: hangi yansıma olduğu. Elenmiş bir yankı
    # yüzünden ardışık olmayabilir, o yüzden dizideki konumdan değil
    # zamanından türetiliyor.
    origin = packets[0]["centroid_time"]
    for packet in packets:
        packet["step"] = (int(round((packet["centroid_time"] - origin)
                                    / coarse[0])) if coarse else
                          packets.index(packet))

    estimates, quality = _pair_delays(values, packets, dt, carrier_hz,
                                      coarse[0] if coarse else None,
                                      thickness_m)
    if not estimates:
        return {"found": False,
                "reason": "Yankılar kapılanamadı — paketler kayıt kenarına "
                          "çok yakın ya da birbirine çok yakın."}

    by_method = {}
    for method in METHODS:
        family = [e["velocity"] for e in estimates if e["method"] == method]
        if family:
            by_method[method] = _summarize(family)

    velocities = [e["velocity"] for e in estimates]
    combined = _summarize(velocities)

    overlap = _overlap_warning(packets, thickness_m,
                               reference_velocity or combined["mean"])
    if overlap:
        warnings.append(overlap)
    spread = _method_warning(by_method)
    if spread:
        warnings.append(spread)
    if quality.get("rejected_pairs"):
        warnings.append(
            "%d yankı çifti, iki yöntem aynı gecikmede anlaşmadığı için "
            "ölçüme katılmadı. Genellikle o yankılardan biri komşusuyla "
            "karışmış ya da ana darbenin kuyruğunda kalmış demektir."
            % quality["rejected_pairs"])
    railed = _clipping_warning(raw_values, packets)
    if railed:
        warnings.append(railed)

    # Kaba (zarf periyodu) ile ince (çevrim zamanlaması) sonuç birbirini
    # tutmalı. Tutmuyorsa çevrim seçimi yanlış bir yere düşmüştür — ve o
    # hata kararlı olduğu için her karede aynı yanlış sayıyı üretir,
    # saçılıma bakarak fark edilemez.
    round_trip = None
    if coarse and combined["mean"]:
        round_trip = coarse[0]
        implied = 2.0 * thickness_m / combined["mean"]
        drift = abs(implied - round_trip) / round_trip
        if drift > ROUND_TRIP_REJECT:
            return {"found": False,
                    "reason": "Zarfın periyodu %s, oysa çevrim zamanlaması "
                              "%s'lik bir gidiş-dönüş veriyor (%%%.0f fark). "
                              "İki yöntem bu kadar ayrıştığında sonuç "
                              "bildirilmiyor. En olası sebep kayıt "
                              "penceresine yeterli yankı sığmaması: ekranda "
                              "ilk paket solda, arkasından 3–4 yankı "
                              "görünmeli."
                              % (_s(round_trip), _s(implied), 100 * drift)}
        if drift > ROUND_TRIP_TOLERANCE:
            warnings.append(
                "Zarfın periyodu %s, oysa hesaplanan hız %s'lik bir gidiş-"
                "dönüş ima ediyor (%%%.0f fark). Bu iki sayı aynı çıkmalı; "
                "farklılarsa yankı kapıları yanlış yere oturmuş ya da "
                "kayıt penceresine yeterli yankı sığmamış olabilir."
                % (_s(round_trip), _s(implied), 100 * drift))
    # Korelasyon tepesinin yüksekliği, sonucun ne kadar güvenilir olduğunun
    # doğrudan ölçüsü: iki yankı gerçekten aynı dalganın gecikmiş kopyasıysa
    # tepe 1'e yakın çıkar. Düştüğünde ya gürültü baskındır ya da
    # kapılanan şey yankı değildir.
    coherence = quality.get("correlation")
    if coherence is not None and coherence < COHERENCE_WARN:
        warnings.append(
            "Yankılar arası korelasyon düşük (%.2f) — kapılanan dalgalar "
            "birbirinin gecikmiş kopyası gibi durmuyor. Donanım ortalamasını "
            "artırın, kazancı yükseltin ya da daha az yankı isteyin."
            % coherence)

    return {
        "found": True,
        "thickness_m": thickness_m,
        "baseline": baseline,
        "sample_interval": dt,
        "carrier_period_s": carrier_s or None,
        "coherence": coherence,
        # Zarftan ölçülen kaba gidiş-dönüş ve onun ima ettiği hız. İnce
        # sonuçtan bağımsız üretildiği için karşılaştırma değeri taşıyor.
        "envelope_round_trip_s": round_trip,
        "envelope_velocity": (2.0 * thickness_m / round_trip
                              if round_trip else None),
        "envelope_correlation": coarse[1] if coarse else None,
        "noise_level": noise,
        "skipped_first_packet": bool(skip_first_packet),
        "packets": packets,
        "estimates": estimates,
        "by_method": by_method,
        "velocity": combined["mean"],
        "velocity_std": combined["std"],
        "velocity_u_a": combined["u_a"],
        "velocity_min": combined["min"],
        "velocity_max": combined["max"],
        "n_estimates": combined["n"],
        "warnings": warnings,
    }


# --- signal conditioning ---------------------------------------------------
def _sample_interval(times):
    if len(times) < 2:
        return None
    return float((times[-1] - times[0]) / (len(times) - 1))


def _carrier_period_samples(values):
    """Taşıyıcı periyodu (örnek) — kaydın en yüksek enerjili bölümünün tayfından.

    Sıfır geçişleri denendi ve iki kez yanıldı: kaydın tamamında sayılınca
    yankılar arası sessizlikteki gürültü periyodu on kat kısaltıyor, genlikle
    kapılanınca da paketin yavaş kısımlarına düşen yarım çevrimler medyanı
    şişiriyor. Tayf, sentetik ve ölçülmüş kayıtların tamamında ikisinden de
    tutarlı çıktı.

    Tayf **kaydın tamamından** değil, en gürültüsüz/en güçlü bölümünden
    alınıyor: bir yankı içermeyen kayıtta uzunluğun neredeyse tamamı taşıyıcı
    taşımıyor ve tümünü dönüştürmek darbenin çizgisini kaydın kendi düşük
    frekanslı biçiminin altına gömüyor.
    """
    return _spectral_period_samples(values)


def _spectral_period_samples(values):
    """Taşıyıcı periyodu — kaydın en gürültülü olmayan bölümünün tayfından."""
    segment = _loudest_stretch(values)
    if segment.size < 16:
        segment = values
    m = segment.size
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(m)))
    first = max(2, m // 2000)
    if spectrum.size <= first + 1:
        return 8.0
    peak_bin = first + int(np.argmax(spectrum[first:]))
    if 0 < peak_bin < spectrum.size - 1:
        y0, y1, y2 = (float(spectrum[peak_bin - 1]), float(spectrum[peak_bin]),
                      float(spectrum[peak_bin + 1]))
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                peak_bin = peak_bin + delta
    if peak_bin <= 0:
        return 8.0
    return max(2.0, float(m) / float(peak_bin))


def _loudest_stretch(values, share=0.25):
    """Kaydin en yuksek enerjili bolumu, taşiyici kestirimi icin.

    Kayan pencereli enerji toplamiyla bulunuyor. Pencere kaydin dortte biri:
    bir yanki dizisini butunuyle icine alacak kadar genis, sessizlikle
    dolmayacak kadar dar.
    """
    n = values.size
    width = max(16, int(n * share))
    if width >= n:
        return values
    power = np.cumsum(np.insert(values ** 2, 0, 0.0))
    energy = power[width:] - power[:-width]
    start = int(np.argmax(energy))
    return values[start:start + width]


def _bandpass(values, dt, centre_hz):
    """Zero-phase band-pass around the probe's centre frequency.

    The pulser dumps a high-voltage spike into a receiver that then has to
    recover, and that recovery is a slow tail riding under the echoes. It
    carries no timing information but it lifts the envelope, so the quiet
    stretch between two echoes never falls back to the floor and the ring-
    down of one echo merges into the next. On the bench this shows up as
    "it only finds the echoes when I raise the high-pass filter, and I have
    to move it again for every thickness".

    Doing it here instead removes the need to chase that knob: the centre
    frequency is measured from the signal, so the band follows whichever
    probe is fitted rather than a setting somebody has to remember.

    The window is real and symmetric in frequency, which makes the filter
    zero-phase. That matters more than the filtering itself -- an ordinary
    causal filter delays the waveform, and while a delay common to every
    echo would cancel in the differences, one that varies with amplitude
    would not.
    """
    n = values.size
    if n < 32 or not dt or not centre_hz:
        return values
    freqs = np.fft.rfftfreq(n, dt)
    low = centre_hz * (1.0 - BANDPASS_WIDTH_RATIO)
    high = centre_hz * (1.0 + BANDPASS_WIDTH_RATIO)
    if high <= low or low >= freqs[-1]:
        return values

    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        return values
    window = np.zeros(freqs.size)
    # Raised cosine rather than a brick wall: a sharp edge in frequency
    # rings in time, and that ringing would look like extra cycles inside
    # every packet -- precisely the thing the cycle picking must not see.
    window[band] = 0.5 * (1.0 - np.cos(2.0 * np.pi
                                       * (freqs[band] - low) / (high - low)))
    return np.fft.irfft(np.fft.rfft(values) * window, n)


def _envelope(values, period_samples):
    """Rectify, hold, then smooth over roughly one carrier period.

    A Hilbert transform would give a cleaner envelope but needs scipy, which
    is not a dependency of this project. A moving maximum over one period
    bridges the zero crossings inside a packet -- which is the only property
    the packet finder actually needs -- and the moving average afterwards
    removes the staircase the hold leaves behind.
    """
    window = max(3, int(round(period_samples)))
    return _moving_mean(_moving_max(np.abs(values), window), window)


def _moving_max(x, w):
    if w <= 1 or x.size < w:
        return x
    pad = w // 2
    padded = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.max(np.lib.stride_tricks.sliding_window_view(padded, w), axis=1)


def _moving_mean(x, w):
    if w <= 1 or x.size < w:
        return x
    pad = w // 2
    padded = np.pad(x, (pad, w - 1 - pad), mode="edge")
    cumulative = np.cumsum(np.insert(padded, 0, 0.0))
    return (cumulative[w:] - cumulative[:-w]) / float(w)


def _envelope_round_trip(envelope, dt, period_samples):
    """Coarse round-trip time from the envelope's autocorrelation.

    Independent of where any packet boundary was drawn: it asks only "at
    what shift does the whole echo train look like itself", which is the
    round trip whatever shape the individual echoes have. That makes it the
    right tool both for sizing the merge gap and, afterwards, for checking
    the fine answer.

    The central lobe is skipped by walking to the first minimum. That lobe
    is the packet correlating with itself; its width is the packet duration,
    and taking the maximum inside it returns a fraction of a microsecond
    rather than a round trip.

    Return: (round trip in seconds, correlation height 0..1), or None.
    """
    if envelope.size < 32 or not dt:
        return None
    # The envelope is already smoothed over a carrier period, so decimating
    # to a few samples per period loses nothing and keeps this affordable on
    # every live frame -- a direct correlation over 45000 points is not.
    step = max(1, int(period_samples / 4.0))
    signal = envelope[::step]
    signal = signal - signal.mean()
    n = signal.size
    if n < 16:
        return None

    size = 1 << int(2 * n - 1).bit_length()
    spectrum = np.fft.rfft(signal, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]

    first = 1
    while first < n - 1 and correlation[first + 1] < correlation[first]:
        first += 1
    limit = n // 2
    if first + 1 >= limit:
        return None

    best = first + int(np.argmax(correlation[first:limit]))
    # A train of echoes correlates at the round trip and at every multiple
    # of it. When a later multiple happens to win, the lag is a whole
    # number of round trips and the velocity comes out an integer fraction
    # of the truth -- so a comparable peak at half the lag is preferred.
    half = best // 2
    if half > first and correlation[half] > 0.6 * correlation[best]:
        best = half
    if correlation[best] < MIN_AUTOCORRELATION:
        return None
    return best * step * dt, float(correlation[best])


def _packets(envelope, period_samples, merge_samples=None, max_samples=None):
    """Contiguous regions of the envelope above threshold, as (start, end).

    Regions closer together than `merge_samples` are merged, because one
    echo's ring-down arrives as several separate humps (see
    PACKET_MERGE_RATIO). Falls back to a carrier period when the round trip
    is not known yet.
    """
    if envelope.size == 0:
        return []
    peak = float(np.max(envelope))
    # The median of the envelope is the quiet floor: packets occupy a few
    # percent of a pulse-echo record, so half of it is gap by a wide margin.
    floor = float(np.median(envelope))
    threshold = max(PACKET_NOISE_FACTOR * floor, PACKET_MIN_RATIO * peak)
    if threshold <= 0 or threshold >= peak:
        return []

    above = (envelope > threshold).astype(np.int8)
    edges = np.diff(above)
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(above) - 1)

    gap = max(2, int(round(merge_samples if merge_samples else period_samples)))
    longest = int(round(max_samples)) if max_samples else None
    merged = []
    for a, b in zip(starts, ends):
        joins = merged and a - merged[-1][1] <= gap
        if joins and longest and b - merged[-1][0] > longest:
            joins = False
        if joins:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    shortest = max(3, int(round(MIN_PACKET_PERIODS * period_samples)))
    return [(a, b) for a, b in merged if b - a + 1 >= shortest]


def _noise_level(values, regions):
    """RMS of the samples that belong to no packet -- the quiet floor."""
    mask = np.ones(values.size, dtype=bool)
    for a, b in regions:
        mask[a:b + 1] = False
    quiet = values[mask]
    if quiet.size < 8:
        return 0.0
    return float(np.sqrt(np.mean(quiet ** 2)))


# --- feature picking -------------------------------------------------------
def _packet_info(times, envelope, a, b, index, period_samples):
    """Bir yankı paketinin konumu ve boyutları.

    Artık paket içinden çevrim seçilmiyor; gecikmeler dalganın tamamından
    korelasyonla çıkarılıyor. Buradan yalnızca kapıyı nereye oturtacağımız
    ve paketin ne kadar geniş olduğu isteniyor.
    """
    window = envelope[a:b + 1]
    envelope_peak = float(np.max(window))
    core_lo, core_hi = _anchor_region(window, envelope_peak)
    centre = _centroid(a + core_lo, window[core_lo:core_hi + 1])
    return {
        "index": index,
        "start_time": float(times[a]),
        "end_time": float(times[b]),
        "start_index": int(a),
        "end_index": int(b),
        "duration": float(times[b] - times[a]),
        # Yarı genlik genişliği: ekonun gerçek gövdesi. `duration`
        # birleştirilmiş bölgenin tamamı ve çınlama kuyruğunu da içeriyor.
        "core_duration": float(times[min(b, a + core_hi)]
                               - times[min(b, a + core_lo)]),
        "envelope_peak": envelope_peak,
        "centroid_index": float(centre),
        "centroid_time": float(np.interp(centre, np.arange(len(times)), times)),
        "peak_amplitude": float(np.max(np.abs(envelope[a:b + 1]))),
        "snr_db": None,
    }


def _keep_on_grid(packets, round_trip, tolerance=GRID_TOLERANCE):
    """Yankı ızgarasına oturmayan paketleri eler.

    Ölçüt paket **ağırlık merkezleri**: ilk pakete göre uzaklık,
    gidiş-dönüşün tam katına `tolerance` yakınlıkta olmalı.

    Dönüş: (kalan paketler, elenen sayısı)
    """
    if not round_trip or len(packets) < 2:
        return packets, 0
    best_kept = []
    # Izgara başlangıcını paket 0'dan veya ana darbe varsa paket 1'den dener
    for start_idx in (0, 1 if len(packets) > 2 else 0):
        origin = packets[start_idx]["centroid_time"]
        candidate = []
        for packet in packets[start_idx:]:
            offset = (packet["centroid_time"] - origin) / round_trip
            if abs(offset - round(offset)) <= tolerance:
                candidate.append(packet)
        if len(candidate) > len(best_kept):
            best_kept = candidate
    if not best_kept:
        best_kept = packets
    return best_kept, len(packets) - len(best_kept)


def _anchor_region(window, envelope_peak):
    """Sub-region of a packet used to place its anchor.

    The run of samples around the envelope maximum that stay above
    `ANCHOR_THRESHOLD_RATIO` of that packet's own peak. Only the run
    containing the maximum is taken: a late ripple that briefly climbs back
    over half amplitude belongs to the ring-down, not to the packet's body,
    and letting it in would drag the anchor later on strong packets only.
    """
    if window.size == 0:
        return 0, 0
    level = ANCHOR_THRESHOLD_RATIO * envelope_peak
    top = int(np.argmax(window))
    lo = top
    while lo > 0 and window[lo - 1] >= level:
        lo -= 1
    hi = top
    while hi < window.size - 1 and window[hi + 1] >= level:
        hi += 1
    return lo, hi


def _centroid(offset, window):
    """Envelope centre of mass of a packet, as a fractional sample index.

    The floor is removed first: the tails of the packet sit on the noise
    level, and leaving them in drags the centroid toward the middle of the
    region rather than the middle of the packet.
    """
    weights = window - float(np.min(window))
    total = float(np.sum(weights))
    if total <= 0:
        return offset + 0.5 * (len(window) - 1)
    positions = np.arange(len(window), dtype=np.float64)
    return offset + float(np.sum(positions * weights) / total)


def _parabolic(times, values, i):
    """Sub-sample position of the extremum at index i.

    Fits a parabola through samples i-1, i, i+1 and returns the position of
    its vertex. The correction is clamped to +/- one sample: a value beyond
    that means the three points are not describing a peak at all (a flat
    region, or noise), and the raw sample is the honest answer there.
    """
    if i <= 0 or i >= len(values) - 1:
        return float(times[i]), float(values[i])

    # Quantisation flattens the top of a cycle into a run of equal samples.
    # A parabola through three equal points is degenerate, and taking the
    # first of them biases every peak to the left by half the run. The
    # midpoint of the run is the unbiased answer.
    left = i
    while left > 0 and values[left - 1] == values[i]:
        left -= 1
    right = i
    while right < len(values) - 1 and values[right + 1] == values[i]:
        right += 1
    if right > left:
        return 0.5 * (float(times[left]) + float(times[right])), float(values[i])

    y0, y1, y2 = float(values[i - 1]), float(values[i]), float(values[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(times[i]), y1
    delta = 0.5 * (y0 - y2) / denom
    if abs(delta) > 1.0:
        return float(times[i]), y1
    step = float(times[i + 1] - times[i])
    return float(times[i]) + delta * step, y1 - 0.25 * (y0 - y2) * delta


# --- velocity --------------------------------------------------------------
def _estimate(i, j, trips, interval, thickness_m, method, strength):
    """Tek bir yankı çiftinden hız kestirimi."""
    if interval is None or interval <= 0:
        return None
    path = 2.0 * trips * thickness_m
    return {
        "from": i + 1,
        "to": j + 1,
        "round_trips": trips,
        "method": method,
        "dt": interval,
        "path_m": path,
        "velocity": path / interval,
        "strength": strength,
    }


def _gate(values, centre, half_width):
    """Bir yankıyı çevresinden koparır.

    Pencere paketin ızgara konumuna oturtuluyor ve kenarları yumuşatılıyor:
    keskin bir kesim, tayfta çınlamaya dönüşür ve faz eğimini bozar. İki
    yankı **aynı genişlikte ve aynı biçimde** kapılandığı için pencere
    ikisini de aynı şekilde etkiler; gecikmeye bir yanlılık taşımaz.

    Kaydın kenarına taşan kısım sıfırla dolduruluyor. Genişliği kısaltmak
    seçenek değil: korelasyon ancak eşit uzunlukta ve aynı ölçekteki
    pencereler arasında anlamlı. Sıfır dolgu zararsız, çünkü Hann penceresi
    zaten uçlarda sıfıra iniyor -- eksik olan kısım pencerelenmiş olsa da
    neredeyse sıfır olacaktı. Yine de yarıdan fazlası eksikse yankı
    kapılanmış sayılmıyor.

    Dönüş: pencere dizisi, ya da kayıttan yeterince veri kalmıyorsa None.
    """
    centre = int(round(centre))
    half_width = int(half_width)
    if half_width < 8:
        return None
    lo, hi = centre - half_width, centre + half_width
    width = hi - lo

    src_lo, src_hi = max(0, lo), min(values.size, hi)
    if src_hi - src_lo < 0.5 * width:
        return None

    gate = np.zeros(width)
    gate[src_lo - lo:src_hi - lo] = values[src_lo:src_hi]
    return gate * np.hanning(width)


def _analytic(x):
    """Analitik sinyal — zarf ve anlık faz için (Hilbert dönüşümü)."""
    n = x.size
    spectrum = np.fft.fft(x)
    mask = np.zeros(n)
    if n % 2 == 0:
        mask[0] = mask[n // 2] = 1.0
        mask[1:n // 2] = 2.0
    else:
        mask[0] = 1.0
        mask[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * mask)


def _correlation_lag(first, second, dt, carrier_hz, max_lag=None):
    """İki yankı arasındaki artık gecikme — zarf kaba, faz ince.

    ASTM C1331'in yöntemi. Korelasyonun **zarfı** tek bir tepe yapar ve
    çevrim belirsizliği taşımaz; ama zarfın tepesi genişçedir, tek başına
    çözünürlüğü sınırlıdır. Tepe noktasındaki **faz** ise çevrimin
    içerisinde nerede olunduğunu söyler. İkisinin birleşimi -- kaba olan
    hangi çevrim, ince olan çevrimin neresi -- bu işin standart çözümü.

    Dönüş: (gecikme saniye, 0..1 arası normalize korelasyon), ya da None.
    """
    n = first.size
    if n != second.size or n < 16 or not carrier_hz:
        return None
    size = 1 << int(2 * n - 1).bit_length()
    spectrum = np.fft.rfft(first, size) * np.conj(np.fft.rfft(second, size))
    circular = np.fft.irfft(spectrum, size)
    # Dairesel korelasyonu -(n-1)..(n-1) gecikmelerine aç.
    correlation = np.concatenate((circular[-(n - 1):], circular[:n]))

    energy = float(np.sqrt(np.sum(first ** 2) * np.sum(second ** 2)))
    if energy <= 0:
        return None

    analytic = _analytic(correlation)
    envelope = np.abs(analytic)

    # Arama, ızgaranın etrafında yarım taşıyıcı çevrimle sınırlanıyor.
    # Dar bantlı bir probda korelasyon zarfı geniştir ve komşu çevrimlerin
    # tepeleri neredeyse eşit yükseklikte olur; sınırsız arama, ölçülen
    # veride bir çifti tam bir çevrim yanlış loba oturttu (o çift %4
    # saparken diğeri %0,1 doğruluktaydı). Kaba ızgara hangi çevrim
    # olduğunu zaten söylüyor; korelasyona düşen iş çevrimin içinde nerede
    # olunduğu.
    centre = n - 1
    if max_lag:
        limit = max(1, int(round(max_lag / dt)))
        lo = max(0, centre - limit)
        hi = min(envelope.size, centre + limit + 1)
    else:
        lo, hi = 0, envelope.size
    peak = lo + int(np.argmax(envelope[lo:hi]))
    strength = float(envelope[peak] / energy)

    lag_samples = peak - centre
    # Zarf tepesindeki faz, çevrim içindeki konumu veriyor. Taşıyıcı
    # periyodunun yarısıyla sınırlı bir düzeltme: daha büyüğü zarfın yanlış
    # çevrimi işaret ettiği anlamına gelir ve düzeltilecek bir şey değildir.
    phase = float(np.angle(analytic[peak]))
    fine = phase / (2.0 * np.pi * carrier_hz)
    return lag_samples * dt - fine, strength


def _phase_slope_lag(first, second, dt, carrier_hz):
    """Çapraz tayfın faz eğiminden gecikme.

    Gecikmiş bir sinyalin tayfı ``e^(-i·ω·τ)`` çarpanı taşır; çapraz tayfın
    fazı bu yüzden frekansla doğrusaldır ve eğimi doğrudan gecikmedir.
    Zamandaki hiçbir noktaya bakmadığı için tepe seçme sorunu hiç doğmuyor.

    Eğim, tayf genliğiyle **ağırlıklandırılarak** uyduruluyor: bandın
    kenarlarında sinyal gürültüye gömülür ve oradaki faz rastgeledir;
    ağırlıksız bir uydurma o gürültüyü ortadaki sağlam bilgiyle eşit sayar.
    """
    n = first.size
    if n != second.size or n < 16 or not carrier_hz:
        return None
    cross = np.fft.rfft(first) * np.conj(np.fft.rfft(second))
    freqs = np.fft.rfftfreq(n, dt)
    band = ((freqs >= carrier_hz * (1.0 - BANDPASS_WIDTH_RATIO))
            & (freqs <= carrier_hz * (1.0 + BANDPASS_WIDTH_RATIO)))
    if int(np.count_nonzero(band)) < 4:
        return None

    weight = np.abs(cross[band])
    total = float(np.sum(weight))
    if total <= 0:
        return None
    weight = weight / total
    omega = 2.0 * np.pi * freqs[band]
    phase = np.unwrap(np.angle(cross[band]))

    omega_mean = float(np.sum(weight * omega))
    phase_mean = float(np.sum(weight * phase))
    spread = float(np.sum(weight * (omega - omega_mean) ** 2))
    if spread <= 0:
        return None
    slope = float(np.sum(weight * (omega - omega_mean)
                         * (phase - phase_mean)) / spread)
    return -slope


def _pair_delays(values, packets, dt, carrier_hz, round_trip, thickness_m):
    """Bütün yankı çiftleri için gecikme ve hız kestirimleri.

    Kapılar paketlerin tespit edilen ağırlık merkezlerine oturtulur.
    İki yankı arasındaki zaman farkı paket merkezleri ve çapraz korelasyon
    ile faz eğimi ince gecikmelerinden türetilir.
    """
    if not round_trip or not dt or not carrier_hz or len(packets) < 2:
        return [], {}
    half = int(round(GATE_HALF_WIDTH_RATIO * round_trip / dt))
    if half < 8:
        return [], {}

    carrier_s = 1.0 / carrier_hz
    max_lag = 0.5 * carrier_s

    # Kapıları her paketin kendi ağırlık merkezine oturt
    gates = {}
    for index, packet in enumerate(packets):
        c_idx = packet["centroid_index"]
        gate = _gate(values, c_idx, half)
        if gate is not None:
            gates[index] = gate

    def pairs(period, gates, consecutive_only):
        found = []
        for i in range(len(packets)):
            for j in range(i + 1, len(packets)):
                if i not in gates or j not in gates:
                    continue
                trips = packets[j]["step"] - packets[i]["step"]
                if trips <= 0 or (consecutive_only and trips != 1):
                    continue
                found.append((i, j, trips, trips * period))
        return found

    # 1. geçiş: ardışık çiftlerden gidiş-dönüşü keskinleştir.
    intervals = []
    for i, j, trips, coarse in pairs(round_trip, gates, True):
        base_dt = packets[j]["centroid_time"] - packets[i]["centroid_time"]
        got = _correlation_lag(gates[i], gates[j], dt, carrier_hz, max_lag)
        if got:
            intervals.append((base_dt - got[0]) / trips)
    if intervals:
        intervals.sort()
        middle = len(intervals) // 2
        round_trip = (intervals[middle] if len(intervals) % 2
                      else 0.5 * (intervals[middle - 1] + intervals[middle]))

    # 2. geçiş: keskinleşmiş gecikmeyle bütün çiftler, iki yöntemle.
    estimates = []
    strengths = []
    rejected = 0
    for i, j, trips, coarse in pairs(round_trip, gates, False):
        base_dt = packets[j]["centroid_time"] - packets[i]["centroid_time"]
        got = _correlation_lag(gates[i], gates[j], dt, carrier_hz, max_lag)
        slope_lag = _phase_slope_lag(gates[i], gates[j], dt, carrier_hz)
        if got is None:
            rejected += 1
            continue

        lag, strength = got
        phase_agrees = (slope_lag is not None
                        and abs(lag - slope_lag) <= PAIR_AGREEMENT_RATIO * carrier_s)

        if not phase_agrees and strength < 0.75:
            rejected += 1
            continue

        strengths.append(strength)
        dt_xcorr = base_dt - lag
        if dt_xcorr > 0:
            estimates.append(_estimate(i, j, trips, dt_xcorr,
                                       thickness_m, METHOD_XCORR, strength))
        if phase_agrees and slope_lag is not None:
            dt_phase = base_dt - slope_lag
            if dt_phase > 0:
                estimates.append(_estimate(i, j, trips, dt_phase,
                                           thickness_m, METHOD_PHASE, None))

    quality = {"correlation": (sum(strengths) / len(strengths))
               if strengths else None,
               "round_trip_s": round_trip,
               "rejected_pairs": rejected}
    return [e for e in estimates if e], quality


def _summarize(velocities):
    values = [float(v) for v in velocities if v is not None]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "u_a": None,
                "min": None, "max": None, "span": None}
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = variance ** 0.5
        u_a = std / (n ** 0.5)
    else:
        std = 0.0
        u_a = None
    return {"n": n, "mean": mean, "std": std, "u_a": u_a,
            "min": min(values), "max": max(values),
            "span": max(values) - min(values)}


# --- warnings --------------------------------------------------------------
def _capture_problem(times, dt, thickness_m, reference_velocity, max_echoes):
    """Yakalama ayarları bu kalınlık için uygun mu.

    Referans hız verilmemişse denetim yapılmıyor: beklenen gidiş-dönüşü
    bilmeden "çok yavaş" ya da "çok az nokta" demenin dayanağı olmaz.
    """
    if not reference_velocity or not dt:
        return None
    expected = 2.0 * thickness_m / float(reference_velocity)
    if expected <= 0:
        return None

    span = float(times[-1] - times[0])
    trips = span / expected
    per_trip = expected / dt

    if per_trip < MIN_SAMPLES_PER_ROUND_TRIP:
        return ("Bir gidiş-dönüşe yalnızca %d örnek düşüyor (%s aralıkla). "
                "Bu çözünürlükte paket içindeki çevrimler görünmez ve prob "
                "frekansı Nyquist'in üstündeyse ölçülen frekans da katlanmış "
                "olur. Kanal başına nokta sayısını artırın ve zaman tabanını "
                "%s civarına çekin (ekranda %d yankı)."
                % (int(per_trip), _s(dt), _s(expected * (max_echoes + 1) / 10.0),
                   max_echoes))

    if trips > max_echoes + MAX_ROUND_TRIPS_IN_RECORD:
        return ("Kayıt penceresine %d gidiş-dönüş sığmış; %d yankı için "
                "gereken %s. Zaman tabanı fazla yavaş: yankılar ekranda üst "
                "üste yığılıyor ve her birine düşen örnek sayısı azalıyor. "
                "Zaman tabanını %s/bölme civarına alın."
                % (int(trips), max_echoes, _s(expected * max_echoes),
                   _s(expected * (max_echoes + 1) / 10.0)))
    return None


def _too_few_packets_reason(regions, times, thickness_m, reference_velocity,
                            round_trip=None):
    """Explains a frame with fewer than two packets.

    "0 packets found" sends the operator looking for a wiring fault. The
    common cause on a thin block is the opposite: the echoes are all there
    but arrive on top of each other, because the packet lasts as long as the
    gap between reflections. That is a property of the probe and the block,
    so the message has to name it -- no setting in the application can fix
    it.
    """
    if not regions:
        return ("Hiç yankı paketi bulunamadı — sinyal yok, kazanç çok düşük "
                "ya da tetikleme kaçıyor olabilir.")

    reference = reference_velocity or REFERENCE_VELOCITY["aluminyum"]
    spacing = round_trip or (2.0 * thickness_m / reference)

    # Kayıt penceresi yetmiyorsa sebep bu; başka bir açıklama aramak
    # operatörü prob ya da kazanç değiştirmeye yönlendirir, oysa yapılması
    # gereken zaman tabanını uzatmak.
    tail = float(times[-1] - times[regions[0][0]])
    if tail < 2.0 * spacing:
        return ("Kayıt penceresine yeterli yankı sığmamış: ilk paketten "
                "sonra %s kalıyor, oysa yankı aralığı yaklaşık %s. En az iki "
                "yankı için zaman tabanını büyütün (ekranda ilk paket solda, "
                "arkasından 3–4 yankı görünmeli)."
                % (_s(tail), _s(spacing)))

    longest = max(float(times[b] - times[a]) for a, b in regions)
    if longest > OVERLAP_RATIO * spacing:
        return ("Yankılar birbirine karışmış: en uzun paket %s sürüyor, oysa "
                "%.3f mm'de yankı aralığı yaklaşık %s. Paket süresi aralığa "
                "yaklaştığı için yansımalar ayrışmıyor — daha yüksek "
                "frekanslı (daha kısa darbeli) prob ya da daha kalın blok "
                "gerekir."
                % (_s(longest), thickness_m * 1000.0, _s(spacing)))
    return ("En az iki yankı gerekiyor, %d paket bulundu. Sonraki yansımalar "
            "gürültünün altında kalmış olabilir — donanım ortalamasını "
            "artırın ya da kazancı yükseltin." % len(regions))


def _overlap_warning(packets, thickness_m, velocity):
    """Warns when the packets are long enough to run into one another.

    Packet duration is set by the probe (cycles / centre frequency) while
    echo spacing is set by the block (2d/c). On a thin block the two become
    comparable and the echoes merge; the packet finder then reports one wide
    region and the interval it measures is meaningless. This is a property
    of the setup, not something the software can correct -- so it is
    reported rather than worked around.
    """
    if not velocity or not packets:
        return None
    spacing = 2.0 * thickness_m / float(velocity)
    longest = max(p.get("core_duration") or p["duration"] for p in packets)
    if longest < OVERLAP_RATIO * spacing:
        return None
    return ("Paket süresi (%s) yankı aralığına (%s) yakın — yankılar üst üste "
            "biniyor olabilir ve seçilen tepe/çukur noktaları yanlış çevrime "
            "düşebilir. Daha yüksek frekanslı prob ya da daha kalın blok "
            "gerekir." % (_s(longest), _s(spacing)))


def _clipping_warning(raw_values, packets):
    """Ölçüme giren paketlerde alıcı doyması var mı.

    Doyma, kaba ve ince sonucun **ikisini birden** aynı yönde bozduğu için
    aralarındaki tutarlılık denetiminden geçip gidiyor: sayılar birbirini
    tutar, tutarlılık yüksek çıkar, sonuç yine de yüzdelerle yanlıştır.
    Dolayısıyla doyma sinyalin kendisinden anlaşılmak zorunda -- düz tepeye
    dayanmış örnek sayısından.
    """
    if raw_values.size == 0 or not packets:
        return None
    high, low = float(np.max(raw_values)), float(np.min(raw_values))
    span = high - low
    if span <= 0:
        return None
    margin = CLIP_RAIL_TOLERANCE * span

    worst = 0.0
    for packet in packets:
        segment = raw_values[packet["start_index"]:packet["end_index"] + 1]
        if segment.size == 0:
            continue
        railed = np.count_nonzero((segment >= high - margin)
                                  | (segment <= low + margin))
        worst = max(worst, railed / float(segment.size))

    if worst < CLIP_WARN_FRACTION:
        return None
    return ("Ölçülen yankıların %%%.0f'ı alıcının sınırına dayanmış — "
            "kazanç yüksek ya da toparlanma kuyruğu sinyali kaydırıyor. "
            "Kırpılmış bir yankının tepe/çukur zamanları kayar ve bunu "
            "hesabın kendisinden anlamak mümkün değildir. Alıcı kazancını "
            "düşürüp tekrarlayın." % (100 * worst))


def _method_warning(by_method):
    """İki yöntem ayrışıyorsa uyarır.

    Çapraz korelasyon ve faz eğimi aynı gecikmeyi tamamen farklı yollardan
    buluyor: biri zaman uzayında dalgaların örtüşmesine, öteki frekans
    uzayında fazın eğimine bakıyor. Ortak bir hataları yok; bu yüzden
    ayrışmaları, tek bir yöntemin kendi içinde tutarlı görünmesinin
    yakalayamayacağı bir sorunu gösterir.
    """
    means = [stats["mean"] for stats in by_method.values() if stats.get("mean")]
    if len(means) < 2:
        return None
    spread = max(means) - min(means)
    average = sum(means) / len(means)
    if average <= 0 or spread / average < 0.01:
        return None
    return ("Çapraz korelasyon ile faz eğimi %%%.2f ayrışıyor (%.0f m/s fark) "
            "— iki bağımsız yöntem aynı sonucu vermiyor, ölçüme güvenmeyin."
            % (100.0 * spread / average, spread))


def uncertainty(result, u_thickness_m, timebase_ppm=25.0, k=2,
                type_a_velocity=None):
    """Bileşik standart ve genişletilmiş belirsizlik hesabı (GUM).

    Hesaplanan belirsizlik bileşenleri:
      * u(d): basamak kalınlığı belirsizliği (operatör girer)
      * u(t): zamanlama belirsizliği — örnekleme aralığı (dikdörtgen dağılım)
              + osiloskop zaman tabanı doğruluğu (ppm)
      * u(A): tekrarlanan karelerden gelen saçılım (tip A). Kare içi
              saçılım bunun yerine kullanılamaz: oradaki kestirimler aynı
              zaman damgalarını paylaşıyor.
    """
    if not result or not result.get("found"):
        return None
    velocity = result.get("velocity")
    thickness = result.get("thickness_m")
    interval = result.get("sample_interval") or 0.0
    if not velocity or not thickness:
        return None

    estimates = result.get("estimates", [])
    if not estimates:
        return None

    shortest = min((e.get("dt", 0.0) for e in estimates if e.get("dt")), default=None)
    if not shortest or shortest <= 0:
        return None

    # Two independent feature times bound each interval, each quantised to
    # the sample grid; a rectangular distribution of width `interval` has
    # standard deviation interval/sqrt(12).
    quantisation = (2.0 ** 0.5) * interval / (12.0 ** 0.5)
    timebase = float(timebase_ppm) * 1e-6 * shortest
    u_time = (quantisation ** 2 + timebase ** 2) ** 0.5

    relative_d = float(u_thickness_m) / thickness if u_thickness_m else 0.0
    relative_t = u_time / shortest

    type_a = type_a_velocity
    if type_a is None:
        type_a = result.get("velocity_u_a") or 0.0
    relative_a = float(type_a) / velocity if velocity else 0.0

    relative = (relative_d ** 2 + relative_t ** 2 + relative_a ** 2) ** 0.5
    standard = relative * velocity
    return {
        "u_relative": relative,
        "u_velocity": standard,
        "k": k,
        "expanded": k * standard,
        "components": {
            "kalinlik": relative_d,
            "zaman_izgarasi": quantisation / shortest,
            "zaman_tabani": timebase / shortest,
            "tip_a": relative_a,
        },
        "dominant": max(
            (("kalinlik", relative_d), ("zaman_izgarasi", quantisation / shortest),
             ("zaman_tabani", timebase / shortest), ("tip_a", relative_a)),
            key=lambda pair: pair[1])[0],
    }


# --- presentation ----------------------------------------------------------
def summary_rows(result, budget=None):
    """(label, value) pairs for the interface and the report."""
    if not result or not result.get("found"):
        return [("Sonuç", result.get("reason", "Yankı bulunamadı") if result else "Yankı bulunamadı")]

    thick = result.get("thickness_m")
    thick_str = ("%.3f mm" % (thick * 1000.0)) if thick is not None else "—"
    vel = result.get("velocity")
    vel_str = ("%.1f m/s" % vel) if vel is not None else "—"
    v_std = result.get("velocity_std")
    v_min = result.get("velocity_min")
    v_max = result.get("velocity_max")
    aralik_str = ("%.1f – %.1f m/s" % (v_min, v_max)) if (v_min is not None and v_max is not None) else "—"

    rows = [
        ("Ses hızı", vel_str),
        ("Blok kalınlığı", thick_str),
        ("Kestirim sayısı", str(result.get("n_estimates", len(result.get("estimates", []))))),
        ("Std sapma", "%.1f m/s" % v_std if v_std is not None else "—"),
        ("Aralık", aralik_str),
        ("Bulunan yankı", str(len(result.get("packets", [])))),
        ("Örnekleme aralığı", _s(result.get("sample_interval"))),
    ]
    if budget:
        rows.append(("Genişletilmiş belirsizlik (k=%d)" % budget["k"],
                     "± %.1f m/s  (%%%.3f)" % (budget["expanded"],
                                               100.0 * budget["u_relative"])))
        rows.append(("Baskın bileşen", budget["dominant"]))
    for method, stats in (result.get("by_method") or {}).items():
        rows.append((METHOD_LABELS[method].capitalize(),
                     "%.1f m/s  (n=%d)" % (stats["mean"], stats["n"])))
    if result.get("envelope_velocity"):
        rows.append(("Zarf periyodundan", "%.1f m/s"
                     % result["envelope_velocity"]))
    if result.get("coherence") is not None:
        rows.append(("Yankı korelasyonu", "%.3f" % result["coherence"]))
    return rows


def estimate_rows(result):
    """One row per velocity estimate -- the detail table on screen."""
    if not result.get("found"):
        return []
    rows = []
    for e in result["estimates"]:
        rows.append({
            "pair": "%d→%d" % (e["from"], e["to"]),
            "feature": METHOD_LABELS[e["method"]],
            "round_trips": e["round_trips"],
            "dt": e["dt"],
            "path_mm": e["path_m"] * 1000.0,
            "velocity": e["velocity"],
            "deviation": (100.0 * (e["velocity"] - result["velocity"])
                          / result["velocity"]) if result["velocity"] else None,
        })
    return rows


def _s(value):
    if value is None:
        return "—"
    for factor, prefix in ((1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")):
        if abs(value) >= factor:
            return "%.4g %ss" % (value / factor, prefix)
    return "%.3g s" % value
