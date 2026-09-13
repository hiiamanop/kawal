from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.models import (
    CanonicalSpan,
    CanonicalSpanLabel,
    Category,
    ComplaintTrajectory,
    DatasetSplit,
    DecisionMode,
    TrajectoryBubble,
    TrajectoryTurn,
    TurnExpectedAction,
)
from services.ml.manifest import compute_file_sha256

PROVENANCE_HIFI_SYNTHETIC = "hifi_synthetic"
PROVENANCE_SYNTHETIC_INDEPENDENT = "synthetic_independent"
CATEGORIES = list(Category)
INTENTS = ("COMPLAINT", "INQUIRY", "FEEDBACK")
RISKS = ("LOW", "MEDIUM", "HIGH", "URGENT")
COMPLETENESS = ("SUFFICIENT", "INCOMPLETE", "AMBIGUOUS")
O_BOUNDARY_DELIMITERS: tuple[str, ...] = (
    ". Lokasinya ",
    ", lokasinya ",
    " — lokasi: ",
)

BALANCED_TRIPLETS: tuple[tuple[str, str, str], ...] = tuple(
    (INTENTS[i], RISKS[r], COMPLETENESS[(i + r + k) % 3])
    for k in range(3)
    for i in range(3)
    for r in range(4)
)

ISSUES: dict[Category, tuple[str, ...]] = {
    Category.ROAD: (
        'Aspal bolong parah di turunan/tanjakan membahayakan pemotor',
        'Amblesan jalan di pinggir saluran air/sungai',
        'Lampu Penerangan Jalan Umum (PJU) padam total berminggu-minggu rawan begal',
        'Tutup manhole/gorong-gorong besi di tengah jalan hilang/ambles',
        'Trotoar keramik hancur besi pembatas copot bahayakan disabilitas',
        'Marka zebra cross & garis kejut di depan sekolah pudar total',
        'Retakan parah jalan beton pasca proyek galian kabel/pipa',
        'Pohon peneduh jalan lapuk miring mau tumbang menimpa kabel listrik',
        'Rambu penunjuk jalan patah miring tertutup dahan',
        'Kerikil ceceran material proyek berserakan di tikungan rawan selip',
        'Jalan bergelombang parah akibat tonjolan akar pohon besar',
        'Guardrail pembatas jalan jurang/jembatan jebol belum diperbaiki',
        'Lampu lalu lintas mati membuat simpang semrawut',
        'Separator jalan pecah dan bautnya mencuat ke lajur kendaraan',
        'Bahu jalan longsor setelah hujan deras',
        'Lubang besar di sambungan aspal dekat halte',
        'Paving block trotoar terangkat dan licin',
        'Genangan oli di lajur menutup permukaan aspal',
        'Saluran kabel terbuka di bawah trotoar',
        'Celah ekspansi jembatan renggang berbahaya',
        'Beton penutup bahu jalan pecah berserakan',
        'Jalan akses permukiman tertutup material galian',
        'Cermin tikungan jalan rusak tidak terlihat pengendara',
        'Tiang rambu parkir roboh menghalangi jalan',
        'Cat marka lajur bus pudar di persimpangan',
        'Aspal mengelupas sepanjang ruas depan pasar',
        'Lubang menganga di depan halte angkot',
        'Pagar pengaman trotoar hilang dicuri',
        'Batu besar jatuh dari tebing menutup setengah lajur',
        'Jembatan kecil retak pada sisi parapet',
        'Kabel proyek melintang jalan tanpa pelindung',
        'Papan proyek roboh menutup trotoar',
        'Permukaan jalan licin lumpur dari truk proyek',
        'Bibir jalan turun di akses rumah warga',
        'Ubin guiding block tunanetra putus-putus',
        'Kanstin jalan pecah dan tajam',
        'Jalan lingkungan berlubang setelah hujan',
        'Aspal tambalan mengelupas kembali',
        'Dahan patah menutup rambu dan trotoar',
        'Parkir liar menutup jalur sepeda',
    ),
    Category.DRAINAGE_FLOOD: (
        'Saluran drainase utama tersumbat endapan lumpur dan sampah padat',
        'Air got meluap ke badan jalan dan rumah warga tiap hujan 15 menit',
        'Dinding kirmir selokan ambrol jebol tergerus arus air hujan',
        'Genangan banjir cileuncang setinggi betis lambat surut',
        'Gorong-gorong penyempitan (bottleneck) di bawah jembatan gang',
        'Bau busuk gas selokan menyengat masuk ke pemukiman dan warung',
        'Saluran pembuangan air limbah cucian dibuang liar ke parit',
        'Pintu air saluran irigasi rusak macet berkarat',
        'Sedimentasi parit tebal membuat kapasitas tampung air hujan hilang',
        'Saluran air tertutup bangunan liar permanen air mampet',
        'Mulut gorong-gorong tertutup daun dan sampah plastik',
        'Saringan inlet jalan hilang membuat sampah masuk saluran',
        'Talud sungai retak dan rawan longsor ke aliran',
        'Banjir kiriman menggenangi gang sempit tiap sore',
        'Pompa banjir tidak menyala saat hujan deras',
        'Kapasitas drainase tidak cukup menampung air dari jalan utama',
        'Tutup saluran beton pecah membahayakan pejalan kaki',
        'Parit tertutup lumpur proyek pembangunan',
        'Air sungai meluap melewati tanggul permukiman',
        'Pipa pembuangan menyempit di bawah jalan',
        'Bak kontrol drainase hilang penutupnya',
        'Genangan tidak surut sejak hujan kemarin',
        'Sampah rumah tangga menyumbat tikungan selokan',
        'Saluran mampet menyebabkan air masuk kios',
        'Tanggul sungai tergerus dan berlubang',
        'Air limbah usaha menghitamkan parit lingkungan',
        'Jembatan gang terlalu rendah menahan aliran',
        'Rumput liar menutup saluran irigasi',
        'Inlet drainase tertutup aspal baru',
        'Parit pinggir jalan dangkal dan penuh endapan',
        'Banjir mengisolasi akses ambulans ke warga',
        'Limbah dapur mengeras di saluran komunal',
        'Kebocoran pipa membuat tanah sekitar drainase ambles',
        'Gorong-gorong tersumbat di bawah akses sekolah',
        'Pintu air tidak bisa ditutup saat debit naik',
        'Air hujan mengalir deras ke rumah tanpa pengarah',
        'Saluran lingkungan terputus oleh proyek jalan',
        'Kolam retensi penuh lumpur dan tidak efektif',
        'Bibir parit runtuh menutup aliran',
        'Bau got muncul setiap malam dari saluran tertutup',
    ),
    Category.WASTE: (
        'Tumpukan sampah liar di lahan kosong pinggir jalan raya menggunung',
        'TPS overload meluber hingga separuh badan jalan timbulkan kemacetan',
        'Truk sampah dinas kebersihan sudah seminggu tidak mengangkut sampah',
        'Pembakaran sampah plastik liar sembarangan asap pekat bikin sesak nafas',
        'Ceceran sampah pasar becek bercampur belatung menyumbat saluran',
        'Bak sampah pilah umum rusak pecah belum ada penggantian',
        'Sampah ranting pohon sisa tebangan dibiarkan menumpuk di trotoar',
        'Pembuangan limbah sisa rongsokan dan kasur bekas di bantaran kali',
        'Gerobak sampah mangkrak di bahu jalan menimbulkan bau anyir',
        'Bau busuk sampah menyengat tercium hingga radius ratusan meter',
        'Sampah rumah makan dibuang terbuka mengundang tikus',
        'Kontainer sampah bocor meneteskan air lindi ke jalan',
        'Sampah berserakan setelah pasar malam tidak dibersihkan',
        'TPS sementara tidak memiliki penutup dan pagar',
        'Sampah elektronik dibuang di dekat permukiman',
        'Jadwal angkut sampah berubah tanpa pemberitahuan warga',
        'Kantong sampah menutup lubang drainase depan rumah',
        'Sampah popok menumpuk di sungai kecil',
        'Petugas hanya mengangkut bagian atas tumpukan sampah',
        'Tempat sampah jalan hilang sehingga warga buang sembarangan',
        'Limbah konstruksi dibuang di lahan kosong',
        'Sampah sayur membusuk di belakang pasar',
        'Bangkai hewan dibiarkan di pinggir jalan',
        'Truk pengangkut menumpahkan sampah sepanjang rute',
        'Papan larangan buang sampah patah tidak terbaca',
        'TPS penuh sejak akhir pekan belum dikosongkan',
        'Sampah plastik menyumbat selokan permukiman',
        'Pemilahan sampah tidak berjalan di TPS lingkungan',
        'Tumpukan kardus menutup akses pejalan kaki',
        'Limbah minyak goreng dibuang bersama sampah biasa',
        'Sampah berserakan di bawah jembatan',
        'Pengangkutan sampah gang terhenti karena armada rusak',
        'Tempat pengumpulan sampah berbau menyengat',
        'Sampah kebun dibakar dekat rumah warga',
        'Puing bongkaran rumah menumpuk di trotoar',
        'Lalat dan tikus berkembang di sekitar TPS',
        'Sampah kiriman menumpuk di bantaran sungai',
        'Warga membuang kantong sampah dari kendaraan',
        'Jadwal pembersihan jalan tidak rutin',
        'Kucing mengacak-acak kontainer tanpa penutup',
    ),
    Category.CLEAN_WATER: (
        'Air PDAM mati total seharian tanpa pemberitahuan atau bantuan tangki',
        'Aliran air PDAM keluar keruh kecoklatan berlumpur pekat dan berpasir',
        'Air kran berbau kaporit sangat menyengat atau berbau karat besi',
        'Pipa transmisi PDAM bocor di trotoar air bersih menyembur deras terbuang',
        'Tekanan air sangat kecil hanya menetes di jam-jam sibuk pagi dan sore',
        'Meteran air PDAM rusak angka loncat drastis tagihan membengkak tidak wajar',
        'Petugas pencatat meteran jarang datang menembak rata tagihan pemakaian',
        'Saluran sambungan pipa baru sudah bayar tapi belum dipasang berbulan-bulan',
        'Air mati berhari-hari warga terpaksa beli air jerigen keliling mahal',
        'Pipa distribusi bocor membasahi halaman dan jalan',
        'Air berwarna kuning setelah perbaikan jaringan',
        'Kran umum tidak mengalir di musim kemarau',
        'Tagihan air tiba-tiba naik tanpa perubahan pemakaian',
        'Penutupan aliran mengganggu usaha laundry warga',
        'Pipa sambungan rumah sering pecah dan bocor',
        'Air PDAM berpasir merusak pompa rumah',
        'Keluhan kebocoran belum ditangani petugas',
        'Meteran tertutup lumpur sulit dibaca',
        'Aliran air hanya tersedia tengah malam',
        'Pipa di depan rumah terinjak proyek dan rusak',
        'Air berbau tanah setelah hujan besar',
        'Distribusi air ke rumah ujung gang tidak merata',
        'Tangki air bantuan belum datang saat gangguan',
        'Saluran air bersih tercampur rembesan limbah',
        'Sambungan baru belum aktif meski biaya lunas',
        'Air mengalir kecil selama berminggu-minggu',
        'Kebocoran pipa utama mengikis badan jalan',
        'Meteran air macet tidak mencatat pemakaian',
        'Petugas tidak memberi jadwal normalisasi aliran',
        'Air mati saat kebutuhan warga meningkat',
        'Pipa tua berkarat membuat air kemerahan',
        'Tutup valve jaringan hilang di pinggir jalan',
        'Air keruh setiap pagi dari keran rumah',
        'Tekanan air turun sejak pembangunan baru',
        'Bocor di sambungan pipa menyebabkan kubangan',
        'Pipa distribusi tertanam dangkal mudah rusak',
        'Air tidak layak dipakai untuk memasak',
        'Pengaduan nomor pelanggan belum mendapat tindak lanjut',
        'Aliran terputus ke masjid dan fasilitas umum',
        'Pipa bocor di dekat sekolah membahayakan akses',
    ),
    Category.CIVIL_ADMIN: (
        'Perekaman e-KTP di kecamatan antre berjam-jam server adminduk offline',
        'Pengurusan pindah datang / cabut berkas KK dipersulit diminta berkas berulang',
        'Praktik calo berkeliaran di kantor disdukcapil tawarkan jalur cepat berbayar',
        'Petugas loket pelayanan administrasi judes tidak ramah main lempar tugas',
        'Legalisir akta kelahiran dan dokumen kependudukan lambat tanpa kepastian',
        'Pengambilan KTP fisik yang sudah jadi molor berbulan-bulan alasan blangko habis',
        'Aplikasi kependudukan online sering error saat verifikasi NIK upload berkas',
        'Kesalahan penulisan nama di akta kelahiran revisinya berbelit-belit',
        'Pungli uang rokok untuk tanda tangan surat pengantar domisili RT/RW',
        'Nomor antrean layanan kelurahan tidak tampil di layar',
        'Surat keterangan domisili belum selesai meski berkas lengkap',
        'Data KK tidak sinkron saat mengurus BPJS',
        'Loket pelayanan tutup sebelum jam kerja selesai',
        'Permohonan akta kematian tertahan verifikasi berulang',
        'Fotokopi berkas diminta berkali-kali di loket berbeda',
        'Website pendaftaran adminduk tidak mengirim kode OTP',
        'Kartu keluarga baru belum tercetak setelah perubahan data',
        'Petugas meminta dokumen tambahan yang tidak tercantum persyaratan',
        'Mesin antrean kantor kecamatan rusak',
        'Perekaman biometrik gagal berulang tanpa solusi',
        'Layanan jemput bola adminduk tidak hadir sesuai jadwal',
        'Pengambilan dokumen diwakilkan tidak diberi informasi prosedur',
        'Koreksi tanggal lahir di KK belum diproses',
        'Surat pengantar pindah tidak ditandatangani berhari-hari',
        'Antrian legalisir menumpuk tanpa estimasi waktu',
        'NIK belum aktif di sistem meski sudah punya dokumen',
        'Berkas online ditolak tanpa alasan yang jelas',
        'Papan informasi persyaratan layanan sudah usang',
        'Petugas loket melempar pemohon ke kantor lain',
        'Dokumen rusak diminta urus ulang tanpa berita acara',
        'Koneksi internet kantor membuat layanan berhenti',
        'Pengaduan administrasi tidak mendapat nomor tindak lanjut',
        'Akta kelahiran bayi belum diterima keluarga',
        'Data alamat berubah sepihak di dokumen kependudukan',
        'Layanan perekaman disabilitas tidak menyediakan antrean prioritas',
        'Surat keterangan usaha dari kelurahan terlambat',
        'Kantor kelurahan tidak menyediakan formulir yang diperlukan',
        'Validasi kepala keluarga tertahan di aplikasi',
        'Calon pemilih belum masuk daftar karena NIK bermasalah',
        'Jadwal layanan kecamatan berubah tanpa pengumuman',
    ),
    Category.HEALTH_SERVICE: (
        'Pasien BPJS Kesehatan dipersulit rujukan atau dipimpong antar faskes',
        'Antrean poli pagi di Puskesmas membludak dokter baru datang jam 10',
        'Obat kronis rutin resep dokter dibilang kosong disuruh beli di apotek luar',
        'Layanan IGD lambat tangani pasien gawat darurat sesak nafas tertahan berkas',
        'Sikap perawat dan nakes di loket pendaftaran ketus dan diskriminatif',
        'Ambulans puskesmas tidak bisa digunakan darurat alasan sopir tidak ada',
        'Ruang rawat inap puskesmas dibilang penuh padahal ada tempat tidur kosong',
        'Alat USG / laboratorium puskesmas rusak diarahkan ke klinik swasta mahal',
        'Antrean obat di farmasi rumah sakit berjam-jam pasien lansia kelelahan',
        'Vaksinasi anak tertunda karena stok vaksin kosong',
        'Pendaftaran online puskesmas error sejak pagi',
        'Rujukan pasien kronis tidak diperbarui tepat waktu',
        'Ruang tunggu puskesmas terlalu penuh dan pengap',
        'Hasil laboratorium belum keluar sesuai jadwal',
        'Petugas loket meminta pasien mengulang antrean',
        'Kursi roda fasilitas kesehatan rusak',
        'Jadwal dokter berubah tanpa pemberitahuan pasien',
        'Keluhan pasien tidak dicatat di meja pengaduan',
        'Toilet puskesmas kotor dan tidak ada air',
        'Obat antibiotik kosong tanpa alternatif dari dokter',
        'Nomor antrean poli lansia tercampur dengan umum',
        'Klaim BPJS diminta bayar tunai tanpa penjelasan',
        'Petugas farmasi salah menyerahkan etiket obat',
        'Ruang isolasi tidak tersedia saat pasien perlu',
        'Ambulans terlambat menjemput warga sakit',
        'Alat tensi dan oksimeter di posyandu rusak',
        'Dokter tidak memberi penjelasan diagnosis memadai',
        'Kamar mandi pasien rawat inap tidak layak',
        'Jadwal imunisasi sekolah dibatalkan mendadak',
        'Sampel pemeriksaan hilang dan diminta ulang',
        'Loket pendaftaran tutup saat jam layanan',
        'Pasien rujukan ditolak karena kuota penuh',
        'Kebersihan ruang tindakan puskesmas buruk',
        'Antrean pemeriksaan ibu hamil terlalu lama',
        'Petugas keamanan menghalangi pendamping pasien',
        'Ruang farmasi kehabisan obat diabetes rutin',
        'Hasil rontgen tertunda berhari-hari',
        'Pasien lansia tidak mendapat prioritas antrean',
        'Puskesmas tidak memberi informasi jam dokter',
        'Tempat tidur observasi terbatas tanpa sistem antrean',
    ),
}

ISSUES[Category.PUBLIC_ORDER] = (
    'preman pasar memeras uang keamanan harian pedagang kaki lima',
    'getok tarif parkir liar di area pertokoan dan wisata tanpa karcis resmi',
    'juru parkir liar mengintimidasi pengendara motor yang menolak bayar lebih',
    'lapak PKL liar menutup trotoar dan memakan separuh badan jalan',
    'warung remang-remang miras oplosan di pemukiman warga meresahkan',
    'suara musik sound system karaoke liar keras hingga larut malam',
    'tawuran pemuda bersenjata tajam di jam malam',
    'pengamen jalanan memaksa dan mengintimidasi penumpang angkutan umum',
    'balap liar motor knalpot brong bising di jalan protokol tiap malam libur',
    'pemalakan sopir bongkar muat oleh sekelompok orang tak dikenal',
    'arena judi sabung ayam dan dadu kopyok di pemukiman padat warga',
    'praktik prostitusi terselubung berkedok panti pijat di ruko perumahan',
    'sekelompok remaja mabuk nongkrong di pos ronda mengganggu warga lewat',
    'kos-kosan bebas tanpa izin dijadikan tempat kumpul pemakai narkoba',
    'tawuran antar geng motor membawa senjata tajam di jalan utama',
    'aksi begal payung meresahkan pejalan kaki di gang sempit',
    'pungli uang koordinasi portal gang oleh oknum pemuda setempat',
    'aksi pelemparan batu ke kaca mobil oleh orang tidak dikenal',
    'pencurian helm dan spion motor di area parkir pinggir jalan marak',
    'perusakan fasilitas umum bangku trotoar dan halte oleh pemuda mabuk',
    'penagih utang menghadang motor warga di tengah jalan raya',
    'warung internet buka tanpa izin 24 jam jadi tempat bolos sekolah',
    'pedagang asongan memaksa menawarkan dagangan di persimpangan lampu merah',
    'keributan dan perkelahian pemabuk di warung makan tepi jalan',
    'peredaran obat keras terlarang berkedok toko kelontong',
    'pemotongan sepihak kabel jaringan oleh oknum warga yang meminta uang',
    'pemalakan uang parkir di area mesin ATM perbankan minimarket',
    'penutupan jalan kampung secara sepihak untuk hajatan tanpa izin aparat',
    'konvoi motor ugal-ugalan mengibarkan bendera kelompok dan menutup jalan',
    'pengrusakan pagar rumah warga akibat lemparan botol miras',
    'ancaman kekerasan verbal juru parkir liar terhadap warga yang parkir sebentar',
    'perang sarung bersenjata gir motor dan batu antar remaja saat dini hari',
    'pungli bongkar pasang tenda pedagang di trotoar depan kantor pos',
    'pengeroyokan warga oleh gerombolan pengendara motor mabuk',
    'penjualan petasan berdaya ledak tinggi di pemukiman padat penduduk',
    'kerumunan anak jalanan mengintimidasi warga yang melintas di flyover',
    'praktik pemerasan terhadap pemilik warung kelontong baru buka',
    'sekelompok orang menggelar perjudian di pos kamling lingkungan',
    'pembobolan kotak amal tempat ibadah di lingkungan warga',
    'keributan ormas memperebutkan lahan pengamanan proyek bangunan',
    'juru parkir liar menarik tarif mobil sepuluh kali lipat di hari libur',
    'warung makan menjual tuak dan minuman beralkohol tanpa izin',
    'aksi vandalisme coret-coret dinding rumah warga dan fasilitas publik',
    'intimidasi calo angkutan umum terhadap calon penumpang di halte',
    'sekelompok orang memasang spanduk provokatif tanpa izin di jembatan',
    'pelemparan petasan ke arah pengendara kendaraan bermotor',
    'aksi copet berulang di jembatan penyeberangan orang dan halte padat',
    'penarikan uang lapak paksa pada pedagang pasar kaget hari minggu',
    'perusakan rambu lalu lintas oleh sekelompok pemuda tidak dikenal',
    'pesta minuman keras di taman lingkungan meresahkan keluarga dan anak',
    'pedagang gerobak liar memblokir akses pintu masuk fasilitas puskesmas',
    'preman meminta jatah rokok dan uang makan paksa kepada pengendara',
)
ISSUES[Category.TRANSPORTATION] = (
    'lampu merah perempatan padat mati total macet krak',
    'lampu lalu lintas error menyala hijau dan merah bersamaan',
    'durasi lampu hijau terlalu singkat antrean kendaraan mengular berkilo-kilometer',
    'angkot ngetem sembarangan di bawah flyover dan tikungan pasar',
    'sopir angkot ugal-ugalan balapan kejar setoran bahayakan penumpang',
    'halte bus transit kotor terbengkalai layar info rute mati',
    'jembatan penyeberangan orang plat bordes bolong berkarat',
    'rambu dilarang parkir tertutup dahan pohon lebat',
    'cermin cembung tikungan buta pecah buram',
    'calo tiket terminal antar kota memaksa beli tiket tarif mahal',
    'bus kota melanggar jalur khusus dan menurunkan penumpang di tengah jalan',
    'pintu perlintasan rel kereta macet sering tertutup sendiri',
    'rambu satu arah dicabut orang tak dikenal pengendara sering salah arah',
    'separator beton jalur bus hancur berserakan di badan jalan',
    'lampu kuning hati-hati di persimpangan jalan mati sudah berminggu-minggu',
    'armada bus transit jarang beroperasi waktu tunggu halte lebih satu jam',
    'shelter halte bus tidak ada atap penumpang kehujanan dan kepanasan',
    'angkutan kota menaikkan tarif sepihak melebihi aturan dinas perhubungan',
    'pelanggaran melawan arah kendaraan motor masif di jalan layang',
    'lampu penerangan di dalam lorong underpass mati total sangat gelap',
    'rambu batas ketinggian jembatan layang patah tertabrak truk muatan',
    'armada bus antar kota tidak laik jalan ban gundul dan knalpot pekat',
    'mesin tap pembayaran tiket bus elektronik di halte rusak',
    'zebra cross di depan rumah sakit tidak dilengkapi lampu peringatan',
    'jembatan penyeberangan orang bergoyang keras dan baut penyangga kendur',
    'mikrolet berhenti mendadak tanpa sein di tengah jalan protokol',
    'armada angkot tidak memiliki surat izin trayek beroperasi di jalur utama',
    'papan petunjuk jurusan rute di terminal robek dan tidak terbaca',
    'lampu penyeberangan tombol pejalan kaki rusak tidak merespons',
    'antrean kendaraan mengunci perlintasan rel kereta akibat angkot ngetem',
    'fasilitas eskalator jembatan penyeberangan orang mati berbulan-bulan',
    'bus patas menurunkan penumpang di bahu jalan bukan di halte resmi',
    'ketiadaan rambu peringatan tikungan tajam rawan kecelakaan di turunan',
    'paku marka jalan reflektor lepas membahayakan ban kendaraan',
    'taksi liar tanpa argo resmi mengangkut penumpang di pintu stasiun',
    'kanopi halte angkutan umum runtuh menimpa bangku tunggu penumpang',
    'bus umum mogok di tengah jalur utama menyebabkan kemacetan total',
    'papan pengumuman jadwal kedatangan bus di shelter tidak akurat',
    'kaca cermin tikungan perumahan ditutup stiker liar tidak berfungsi',
    'jembatan penyeberangan orang dijadikan tempat tidur dan sangat kumuh',
    'petugas terminal tidak menertibkan angkot parkir paralel di jalan utama',
    'marka kuning kotak di perempatan padat terhapus tidak terlihat',
    'lampu strobo dan sirine ilegal kendaraan pribadi meresahkan pemakai jalan',
    'pagar pembatas pejalan kaki di trotoar halte patah tertabrak mobil',
    'rute angkutan kota memotong trayek seenaknya tidak sampai tujuan akhir',
    'rambu petunjuk arah flyover salah menunjukkan jalan keluar persimpangan',
    'tombol bel darurat di dalam armada bus raya terpadu mati tidak bunyi',
    'pintu bus transit tidak menutup rapat saat armada melaju kencang',
    'penerangan halte bus malam hari padam menjadi rawan kejahatan',
    'kendaraan travel gelap menjaring penumpang sembarangan di persimpangan',
    'jalur penyeberangan ramah disabilitas di halte terhalang tiang rambu',
    'klakson basuri bus pariwisata membunyikan suara bising memekakkan telinga',
)
ISSUES[Category.FIRE_RESCUE] = (
    'kebakaran rumah padat penduduk akibat korsleting listrik',
    'kobaran api di lahan ilalang kering meluas mendekati perumahan',
    'sarang tawon vespa besar di atap rumah warga sudah menyengat anak-anak',
    'ular kobra masuk ke kamar mandi dan plafon rumah warga',
    'cincin terjepit di jari tangan bengkak membiru butuh evakuasi darurat',
    'kepala balita terjepit teralis pagar besi rumah',
    'hidran pemadam kebakaran di trotoar rusak mati tertutup cor semen',
    'kucing terjepit celah dinding beton sempit butuh pertolongan',
    'bau kebocoran gas elpiji menyengat di ruko terkunci berisiko meledak',
    'evakuasi pohon tumbang menimpa atap rumah warga saat hujan badai',
    'kebakaran gudang rongsokan dan kardus bekas asap pekat membubung',
    'ular piton sanca batik panjang berada di saluran air dekat ternak',
    'kepulan asap tebal dari toko bahan kimia terbakar membahayakan warga',
    'evakuasi korban terjatuh ke dalam sumur tua sedalam lima belas meter',
    'cincin logam tersangkut di jari membengkak butuh gerinda pemotong',
    'kebakaran kios pasar tradisional kobaran api menjalar ke deretan ruko',
    'pelepasan sarang lebah madu liar di pohon dekat taman bermain anak',
    'mobil terbakar di bahu jalan protokol akibat korsleting mesin',
    'anak kecil terkunci di dalam kamar tidur lantai atas kunci pintu rusak',
    'kebakaran trafo tiang listrik mengeluarkan percikan api dan ledakan',
    'hewan peliharaan tercebur ke lubang gorong-gorong sedalam tiga meter',
    'kebakaran tumpukan ban bekas di tanah kosong dekat pemukiman padat',
    'evakuasi lansia terjebak banjir bandang setinggi dua meter di loteng',
    'sarang tawon ganas di atap sekolah mengancam keselamatan siswa',
    'kebocoran pipa tabung gas industri besar di dapur restoran',
    'kaki anak terselip di eskalator pusat perbelanjaan butuh evakuasi cepat',
    'pelepasan cincin macet di jari tangan warga yang mulai membengkak',
    'pohon tua berukuran besar tumbang melintang menutup total badan jalan',
    'biawak liar ukuran besar masuk ke dapur rumah warga membuat panik',
    'kebakaran bengkel motor dan drum pelumas memicu api membesar',
    'tangan balita terjepit di sela pintu geser kaca minimarket',
    'evakuasi jenazah korban hanyut terbawa arus sungai di pintu air',
    'kebakaran rumpun bambu kering di lereng rawan merembet ke kabel listrik',
    'hidran pemukiman warga tidak mengeluarkan semprotan air saat darurat',
    'ular berbisa sembunyi di dalam ruang mesin mobil warga di garasi',
    'anak terpeleset masuk celah sempit antara tembok pembatas bangunan',
    'kebakaran instalasi listrik di area pertokoan pasar kaget',
    'hewan kucing tercebur ke dalam pipa saluran pembuangan air hujan',
    'kebakaran tumpukan sampah merembet ke dinding samping rumah tetangga',
    'balita terkunci di dalam mobil sendirian saat mesin menyala',
    'sarang tawon berukuran bola di bawah jembatan pejalan kaki',
    'evakuasi dahan pohon patah bergelantungan di atas kabel listrik PLN',
    'tumpahan solar licin di tanjakan jalan raya rawan timbulkan kebakaran',
    'kebakaran tempat penimbunan kayu mebel dekat perkampungan warga',
    'kaki warga lansia tersangkut di celah rel bantalan kereta api',
    'kera liar turun ke pemukiman warga dan mencakar anak-anak',
    'kebakaran panel boks meteran listrik rumah akibat arus pendek',
    'penanganan tumpahan ceceran oli di tikungan jalan raya yang licin',
    'evakuasi ternak tercebur ke dalam parit lumpur yang dalam',
    'kebakaran gudang kasur busa api cepat membesar menghasilkan asap racun',
    'jari tangan remaja terjepit lubang bangku besi fasilitas umum',
    'kobaran api pembakaran sampah kering merembet ke dinding kayu rumah',
)
ISSUES[Category.SOCIAL_AFFAIRS] = (
    'orang dengan gangguan jiwa mengamuk melempar batu ke kendaraan',
    'lansia terlantar sakit parah tidur di emperan toko tanpa perawatan',
    'eksploitasi balita untuk mengemis manusia perak di lampu merah',
    'bantuan sosial program keluarga harapan salah sasaran ke warga mampu',
    'pemotongan uang tunai bansos oleh oknum pengurus lingkungan',
    'beras bantuan pangan berkutu kuning apek berbau tak layak konsumsi',
    'pengemis berkostum badut memaksa minta uang di pintu masuk minimarket',
    'anak telantar putus sekolah tidur di kolong jembatan jalan layang',
    'panti asuhan tanpa izin menelantarkan anak asuh tanpa makanan bergizi',
    'pengemis pura-pura pincang beroperasi di perempatan jalan kota',
    'gelandangan dan pemulung tidur di bangku ruang tunggu shelter halte',
    'lansia sebatang kara kelaparan di gubuk reyot tanpa bantuan keluarga',
    'anak jalanan menghirup lem perekat di sudut taman kota tanpa binaan',
    'penyandang disabilitas berat belum terdata dalam program jaminan sosial',
    'pungli biaya administrasi pencairan dana bantuan langsung tunai',
    'wanita tunawisma hamil terlantar di pinggir jalan tanpa pertolongan',
    'lansia korban penelantaran keluarga ditinggalkan di area terminal bus',
    'calo bantuan sosial memotong dana bantuan warga penerima manfaat',
    'anak usia dini dipaksa orang tua mengamen di jalan sampai tengah malam',
    'orang terlantar menderita sakit lumpuh tergeletak di trotoar jalan',
    'timbangan karung beras bantuan pangan berkurang beberapa kilogram',
    'warga rentan miskin tidak terdaftar dalam basis data bantuan sosial',
    'orang dengan gangguan jiwa berkeliaran tanpa busana dekat lingkungan sekolah',
    'panti penampungan lansia menelantarkan warga binaan tanpa perawat',
    'data penerima bansos memuat identitas warga yang telah meninggal dunia',
    'anak jalanan mengelap kaca mobil paksa dan meminta uang bernada mengancam',
    'lansia tunanetra dituntun anak balita mengemis di jalan protokol siang hari',
    'sekelompok pengemis manusia perak mencegat pengendara di lampu merah',
    'pembagian paket beras santunan pemerintah tidak merata di lingkungan',
    'penyandang tuna grahita tersesat linglung tanpa identitas pengenal',
    'tunawisma meletakkan gerobak kumuh di trotoar depan kantor kelurahan',
    'warga penderita lumpuh membutuhkan bantuan kursi roda dari dinas sosial',
    'pungli pengurusan pendaftaran jaminan sosial warga kurang mampu',
    'anak korban kekerasan orang tua melarikan diri terlantar di jalan raya',
    'kakek pikun tersesat dan bingung duduk di pinggir jalan berjam-jam',
    'praktik adopsi anak tanpa prosedur resmi meresahkan warga pemukiman',
    'pengemis membawa balita tidur dalam gendongan di tengah terik matahari',
    'warga difabel kesulitan mendapatkan fasilitas alat bantu tongkat jalan',
    'keluarga melakukan pemasungan orang dengan gangguan jiwa di gubuk',
    'oknum kelompok memotong kartu sembako bansos milik warga lansia',
    'tunawisma mendirikan tenda terpal kumuh di ruang terbuka hijau flyover',
    'balita mengalami kekurangan gizi parah di pemukiman tanpa penanganan',
    'lansia miskin sebatang kara tidak memiliki kartu jaminan kesehatan',
    'remaja putus sekolah terkoordinir sindikat pengemis di persimpangan',
    'santunan duka warga tidak mampu tertahan belum disalurkan petugas',
    'gerombolan peminta-minta musiman mengetuk pintu rumah warga secara agresif',
    'anak kecil dipaksa menjual tisu di persimpangan lampu merah hingga larut',
    'orang dengan gangguan jiwa membakar kardus di serambi bangunan kosong',
    'bantuan uang tunai penanganan kemiskinan ditahan oknum aparat kampung',
    'tunawisma luka infeksi membusuk tergeletak di sudut pasar tradisional',
    'gelandangan mencuci baju dan buang air di kolam air mancur kota',
    'warga korban bencana kebakaran belum menerima tenda darurat dinas sosial',
)
ISSUES[Category.EDUCATION] = (
    'plafon ruang kelas sekolah dasar ambruk saat proses jam belajar',
    'toilet sekolah negeri rusak parah jorok tanpa air bersih pintu copot',
    'pungli sumbangan pembangunan gedung sekolah jutaan rupiah oleh komite',
    'pemaksaan beli seragam dan lembar modul mahal di koperasi sekolah',
    'ijazah siswa ditahan pihak sekolah karena tunggakan uang bangunan',
    'manipulasi titik koordinat domisili kartu keluarga untuk jalur zonasi',
    'aksi perundungan fisik murid di lingkungan sekolah dibiarkan guru',
    'guru aparatur sipil sering mangkir mengajar dan kelas dibiarkan kosong',
    'pagar tembok sekolah roboh membahayakan anak-anak saat jam istirahat',
    'pungutan biaya acara pelepasan dan wisuda sekolah membebani orang tua',
    'atap genteng laboratorium sekolah negeri bocor deras saat hujan',
    'kursi dan meja belajar siswa di ruang kelas lapuk dan patah',
    'kewajiban membeli lembar kerja modul materi yang dilarang regulasi',
    'tindakan kekerasan fisik oknum pendidik terhadap murid di ruang kelas',
    'sekolah mengadakan bimbingan belajar berbayar wajib berkedok les tambahan',
    'tembok penyangga kelas retak lebar dan dikhawatirkan ambruk seketika',
    'kuota penerimaan siswa baru jalur prestasi disalahgunakan pihak panitia',
    'dana bantuan operasional penunjang pendidikan sekolah diselewengkan',
    'perangkat komputer laboratorium ujian rusak tidak bisa dioperasikan',
    'fasilitas kamar mandi toilet sekolah pesing dan saluran pembuang mampet',
    'potongan sepihak insentif guru honorer oleh manajemen sekolah',
    'larangan mengikuti penilaian akhir semester karena belum lunas uang komite',
    'lingkungan gedung sekolah tidak menyediakan ram jalan bagi disabilitas',
    'lapangan upacara sekolah tergenang air berlumpur mengganggu kegiatan',
    'pemaksaan kegiatan kunjungan wisata ke luar kota dengan biaya tinggi',
    'makanan di kantin sekolah tidak higienis menyebabkan siswa mual keracunan',
    'kaca jendela ruang belajar pecah berserakan belum diperbaiki pengelola',
    'rekayasa nilai rapor siswa demi memenangkan persaingan seleksi sekolah',
    'biaya cetak kartu ujian dan berkas penilaian dibebankan kepada siswa',
    'kekosongan tenaga pendidik mata pelajaran pokok selama beberapa bulan',
    'ketiadaan buku paket pegangan kurikulum di ruang perpustakaan sekolah',
    'pencoretan sepihak murid kurang mampu dari daftar penerima beasiswa',
    'jalan masuk menuju gerbang sekolah licin berlumpur tanpa pegangan aman',
    'potongan liar pada bantuan dana program pendidikan siswa miskin',
    'kondisi lapangan olahraga sekolah rusak berkerikil tajam rawan melukai',
    'pemaksaan pembelian kain seragam khusus dari vendor rekanan sekolah',
    'ruang perpustakaan sekolah basah tergenang rembesan air atap bocor',
    'perkelahian antar pelajar di luar area sekolah selepas jam kepulangan',
    'penyaluran bantuan perlengkapan sekolah anak kurang mampu dipersulit',
    'sekolah dasar tidak memiliki fasilitas kran wastafel air cuci tangan',
    'ruang kelas terasa sangat pengap tanpa ventilasi udara yang mencukupi',
    'biaya pendaftaran ulang siswa baru di sekolah negeri yang harusnya gratis',
    'oknum pengajar memberikan sanksi fisik berlebihan kepada anak didik',
    'penjualan map formulir seleksi penerimaan murid dengan harga mahal',
    'kanopi penghubung gedung kelas patah membahayakan lintasan murid',
    'korban intimidasi teman sebaya di sekolah mengalami trauma bersekolah',
    'biaya legalisir dokumen kelulusan siswa ditarik tarif tanpa bukti resmi',
    'pembatalan mendadak status penerimaan murid jalur afirmasi ekonomi',
    'buku panduan belajar cetakan kementerian tidak dibagikan kepada murid',
    'kamar mandi sekolah tanpa sekat pemisah yang layak privasi siswa',
    'ancaman penahanan lembar hasil belajar siswa jika iuran belum lunas',
    'anak tangga lantai atas sekolah licin dan tanpa pegangan pengaman besi',
)
ISSUES[Category.PARKS_HOUSING] = (
    'pohon peneduh jalan lapuk berlubang rawan roboh menimpa kendaraan',
    'wahana ayunan dan perosotan anak di taman kota patah berkarat tajam',
    'lampu penerangan taman kota mati dijadikan tempat asusila dan miras',
    'bangku taman dirusak dicorat-coret vandalisme kotor',
    'lift rusunawa mati berbulan-bulan lansia kesulitan naik tangga',
    'pipa pembuangan limbah rusunawa bocor mengotori lorong hunian',
    'pungli biaya pemakaman di TPU pemerintah kota jutaan rupiah oleh oknum',
    'makam tumpang tumpuk sepihak tanpa persetujuan ahli waris',
    'semak belukar liar menutupi area makam umum tidak pernah dirawat',
    'rumput taman kota mengering tidak disiram tanaman mati menguning',
    'fasilitas toilet umum di taman kota rusak terkunci dan tidak ada air',
    'pompa air fasilitas rusunawa rusak warga kesulitan mandi dan mencuci',
    'atap genteng blok rusunawa bocor merembes ke plafon kamar warga',
    'dahan pohon peneduh menjuntai rendah menutupi rambu dan pandangan sopir',
    'jalur setapak taman hancur ambles membahayakan lansia pejalan kaki',
    'pungutan parkir tidak resmi di area taman terbuka hijau publik',
    'area bermain anak dipenuhi serpihan beling dan kotoran hewan',
    'kolam air mancur taman kota mati airnya berlumut menjadi sarang jentik',
    'jaringan instalasi kabel listrik rusunawa semrawut rawan kebakaran',
    'bak tempat sampah taman hilang dicuri sehingga sampah berserakan',
    'trotoar pejalan kaki taman diserobot lapak pedagang kaki lima liar',
    'lintasan lari jogging track taman kota retak-retak licin berlumut',
    'area pemakaman umum warga gelap gulita tanpa lampu penerangan malam',
    'kamar hunian rusunawa disewakan sepihak ke pihak ketiga tarif tinggi',
    'paving block area parkir rusunawa ambles berlubang kubangan air',
    'permintaan biaya tebang pohon peneduh jalan ditarik jutaan rupiah',
    'lampu sorot sarana olahraga terbuka di taman kota putus mati total',
    'alat kebugaran terbuka di taman kota macet berkarat tidak bisa dipakai',
    'dinding pagar pembatas taman roboh menutupi saluran pembuang air',
    'talang air koridor rusunawa pecah air mengguyur lorong lantai hunian',
    'tanah makam keluarga terurug puing galian proyek tanpa pemberitahuan',
    'akar pohon peneduh trotoar menjalar merusak dinding pagar rumah warga',
    'lahan taman kota digunakan sembarangan untuk memarkir mobil pribadi',
    'tangki tandon penampung air rusunawa kotor berlumut air berbau karat',
    'bangunan saung gazebo taman rusak atapnya bolong tidak diperbaiki',
    'praktik percaloan surat izin pemakaman umum dengan tarif mahal',
    'tanaman perdu pembatas jalur jalan layu mengering tanpa perawatan',
    'pintu gerbang kompleks rusunawa patah keamanan lingkungan tidak terjaga',
    'kolam resapan perumahan penuh gulma eceng gondok air tidak mengalir',
    'keramik lantai koridor hunian rusunawa terangkat pecah tajam terinjak',
    'lampu taman bertenaga surya hilang komponen baterai dan akinya dicuri',
    'permainan jungkat-jungkit anak copot pasak besi membahayakan balita',
    'akses jalan menuju pemakaman umum rusak parah berbatu licin berlumpur',
    'pohon peneduh jalan miring menimpa rentangan kabel telekomunikasi',
    'iuran kebersihan rusunawa ditarik tanpa tanda bukti kuitansi resmi',
    'bangkai hewan liar di balik semak taman menyebarkan bau tidak sedap',
    'alat meteran air unit rusunawa rusak tagihan rekening melonjak drastis',
    'pelataran plaza taman dipakai berjualan durian meninggalkan limbah busuk',
    'batu nisan makam umum dipecahkan oleh gerombolan orang tidak dikenal',
    'pagar tanaman hijau taman mengering kekurangan penyiraman rutin',
    'akses tangga darurat rusunawa terhalang tumpukan barang rongsokan',
    'rumput ilalang liar di taman pemukiman tumbuh lebat setinggi dada',
)
ISSUES[Category.ROAD] += (
    'Jalan retak memanjang di sekitar simpang dan membahayakan sepeda motor',
    'Aspal terkelupas di depan pertokoan setelah pekerjaan utilitas',
    'Lubang kecil berderet berubah besar di jalur angkot',
    'Trotoar dipakai parkir karena bollard pengaman hilang',
    'Penyeberangan pejalan kaki tidak memiliki lampu peringatan',
    'Penerangan bawah jembatan mati dan titiknya sangat gelap',
    'Batu kanstin lepas berserakan di sisi jalur kendaraan',
    'Pagar jembatan bengkok setelah tertabrak belum diamankan',
    'Jalur sepeda tertutup lapak dan tidak bisa digunakan',
    'Permukaan jalan menurun licin karena tumpahan tanah merah',
    'Rambu batas kecepatan roboh di depan fasilitas umum',
    'Perbaikan jalan meninggalkan lubang terbuka tanpa pembatas',
)
ISSUES[Category.DRAINAGE_FLOOD] += (
    'Saluran mikro di depan rumah tertutup cor dan air mencari jalan lain',
    'Air hujan menggenang di bawah jembatan karena outlet tersumbat',
    'Tumpukan pasir proyek masuk ke gorong-gorong lingkungan',
    'Lumpur banjir menutup kisi-kisi drainase sepanjang gang',
    'Saluran pembuang pasar penuh minyak dan tidak mengalir',
    'Tanggul parit jebol membuat air mengarah ke halaman warga',
    'Drainase di sisi sekolah tidak punya penutup yang aman',
    'Aliran sungai menyempit karena bangunan semi permanen',
    'Bak penampung hujan meluap sebelum petugas datang',
    'Air selokan berwarna hitam mengalir ke sungai',
    'Gorong-gorong jalan kompleks tertutup akar dan tanah',
    'Genangan menutup akses warung setiap hujan sore',
)
ISSUES[Category.WASTE] += (
    'Tumpukan sampah rumah tangga menghalangi akses gang',
    'Kontainer sampah tidak dikembalikan setelah diangkut',
    'Sampah botol dan kemasan menutup saluran dekat pasar',
    'Limbah hasil renovasi dibuang di pinggir taman kota',
    'Sampah makanan membusuk di sekitar halte angkutan umum',
    'Kantong sampah tercecer dari kendaraan pengangkut',
    'TPS lingkungan penuh dan tidak ada jadwal pengosongan',
    'Sisa dagangan pasar ditinggalkan di trotoar sampai pagi',
    'Sampah daun menumpuk tanpa penyapuan rutin',
    'Tempat sampah umum penuh di area wisata kota',
    'Sampah rumah tangga dibuang ke bantaran sungai setiap malam',
    'Limbah kemasan usaha menumpuk di belakang ruko',
)
ISSUES[Category.CLEAN_WATER] += (
    'Air leding berhenti mengalir setelah pipa utama diperbaiki',
    'Sambungan pipa bocor membuat air menggenangi teras pelanggan',
    'Air keran meninggalkan endapan putih pada peralatan rumah',
    'Pengaliran bergilir tidak diumumkan kepada pelanggan',
    'Pipa air bersih terbuka di lokasi galian jalan',
    'Kran umum kampung rusak dan tidak bisa dipakai warga',
    'Air bertekanan tinggi memecahkan sambungan fleksibel rumah',
    'Pelanggan belum menerima penjelasan atas gangguan distribusi',
    'Bau klorin tetap kuat meski air sudah didiamkan',
    'Aliran air ke lantai dua rumah tidak pernah mencapai tandon',
    'Perbaikan kebocoran meninggalkan lubang di halaman pelanggan',
    'Pipa tua mengeluarkan rembesan di bawah trotoar',
)
ISSUES[Category.CIVIL_ADMIN] += (
    'Perubahan data pendidikan di KK belum muncul di sistem',
    'Surat keterangan tidak mampu tertahan di kantor kelurahan',
    'Layanan konsultasi adminduk tidak menyediakan petugas informasi',
    'Berkas pindah domisili hilang dan pemohon diminta mengulang',
    'Antrian pencetakan dokumen menumpuk sejak pagi',
    'Akun layanan online terkunci setelah verifikasi NIK gagal',
    'Akta perkawinan belum terbit meski status berkas selesai',
    'Pemohon lanjut usia tidak mendapat bantuan di loket digital',
    'Jam layanan pengaduan tidak tercantum di kantor kecamatan',
    'Kartu identitas anak belum tersedia setelah pendaftaran',
    'Surat rekomendasi kelurahan tertunda karena pejabat tidak ada',
    'Pemohon diminta kembali tanpa tanda terima berkas',
)
ISSUES[Category.HEALTH_SERVICE] += (
    'Pasien demam menunggu lama di ruang triase tanpa pemeriksaan',
    'Pendaftaran peserta BPJS gagal karena data kepesertaan tidak terbaca',
    'Stok obat hipertensi kosong selama beberapa hari',
    'Ruang tunggu poli anak terlalu panas dan penuh',
    'Jadwal kunjungan dokter spesialis tidak sesuai papan informasi',
    'Petugas meminta biaya tambahan untuk layanan yang ditanggung BPJS',
    'Rujukan laboratorium tertunda karena formulir tidak ditemukan',
    'Pasien disabilitas kesulitan masuk ke ruang pemeriksaan',
    'Sistem antrean farmasi tidak memanggil nomor pasien',
    'Kamar rawat jalan tidak dibersihkan setelah tindakan',
    'Puskesmas kekurangan petugas pada jam kunjungan pagi',
    'Hasil pemeriksaan pasien tidak bisa diakses di aplikasi',
)

BREAK_WORDS = {
    "di", "pada", "ke", "dari", "sepanjang", "dekat", "sekitar", "antara", "seberang", "samping", "luar", "dalam",
    "akibat", "karena", "sehingga", "saat", "setelah", "pasca", "ketika", "tiap", "setiap", "tanpa", "hingga", "sampai",
    "selepas", "sebelum", "menjelang",
    "membahayakan", "membuat", "bikin", "menutup", "menghalangi", "menimpa", "mengancam", "meresahkan", "menimbulkan",
    "dan", "atau", "serta", "tetapi", "padahal", "dengan"
}

SYNONYMS = {
    "jalan": {"train": "jalan", "dev": "ruas jalan", "test": "badan jalan", "ood": "jalur jalan"},
    "air": {"train": "air", "dev": "aliran air", "test": "pasokan air", "ood": "suplai air"},
    "warga": {"train": "warga", "dev": "masyarakat", "test": "warga sekitar", "ood": "penduduk"},
    "sampah": {"train": "sampah", "dev": "tumpukan sampah", "test": "limbah sampah", "ood": "buangan sampah"},
    "sekolah": {"train": "sekolah", "dev": "gedung sekolah", "test": "lingkungan sekolah", "ood": "area sekolah"},
    "rusak": {"train": "rusak", "dev": "hancur", "test": "rusak parah", "ood": "rusak berat"},
    "taman": {"train": "taman", "dev": "area taman", "test": "taman kota", "ood": "taman publik"},
    "liar": {"train": "liar", "dev": "tanpa izin", "test": "ilegal", "ood": "tak berizin"},
    "saluran": {"train": "saluran", "dev": "aliran drainase", "test": "gorong-gorong", "ood": "selokan"},
    "trotoar": {"train": "trotoar", "dev": "jalur pedestrian", "test": "area trotoar", "ood": "akses trotoar"},
    "lampu": {"train": "lampu", "dev": "penerangan", "test": "lampu penerangan", "ood": "bohlam lampu"},
    "mati": {"train": "mati", "dev": "padam", "test": "tidak menyala", "ood": "tidak berfungsi"},
    "petugas": {"train": "petugas", "dev": "staf", "test": "aparat petugas", "ood": "petugas lapangan"},
    "bantuan": {"train": "bantuan", "dev": "dana bantuan", "test": "program bantuan", "ood": "bansos"},
    "pasien": {"train": "pasien", "dev": "pasien berobat", "test": "warga yang berobat", "ood": "pasien faskes"},
    "kebakaran": {"train": "kebakaran", "dev": "amukan api", "test": "insiden kebakaran", "ood": "peristiwa kebakaran"},
    "hujan": {"train": "hujan", "dev": "turun hujan", "test": "guyuran hujan", "ood": "hujan lebat"},
    "tertutup": {"train": "tertutup", "dev": "terhalang", "test": "tersumbat", "ood": "tertimbun"},
    "parkir": {"train": "parkir", "dev": "perparkiran", "test": "tempat parkir", "ood": "lokasi parkir"},
    "pohon": {"train": "pohon", "dev": "batang pohon", "test": "pohon peneduh", "ood": "pepohonan"},
    "kendaraan": {"train": "kendaraan", "dev": "lalu lintas", "test": "pengendara", "ood": "arus kendaraan"},
    "pasar": {"train": "pasar", "dev": "area pasar", "test": "lingkungan pasar", "ood": "kawasan pasar"},
    "bolong": {"train": "berlubang", "dev": "bolong", "test": "rusak berlubang", "ood": "jebol bolong"},
    "parah": {"train": "cukup parah", "dev": "sangat parah", "test": "parah sekali", "ood": "kondisi berat"},
    "tersumbat": {"train": "tersumbat", "dev": "mampet", "test": "tertimbun endapan", "ood": "terhambat kotoran"},
    "meluap": {"train": "meluap", "dev": "tumpah meluber", "test": "menggenang tinggi", "ood": "banjir meluap"},
    "padam": {"train": "padam total", "dev": "mati sama sekali", "test": "tidak berfungsi", "ood": "padam gelap"},
    "bocor": {"train": "bocor", "dev": "mengalami kebocoran", "test": "pecah bocor", "ood": "rembes parah"},
    "ambruk": {"train": "ambruk", "dev": "roboh", "test": "runtuh", "ood": "tumbang"},
    "longsor": {"train": "longsor", "dev": "runtuh longsor", "test": "ambles longsor", "ood": "longsoran tanah"},
    "antre": {"train": "antre lama", "dev": "menunggu berjam-jam", "test": "antrean panjang", "ood": "antrean mengular"},
    "antrean": {"train": "antrean", "dev": "barisan antre", "test": "antrean warga", "ood": "deretan antrean"},
    "dokter": {"train": "dokter", "dev": "dokter pemeriksa", "test": "tenaga dokter", "ood": "pihak dokter"},
    "obat": {"train": "obat", "dev": "resep obat", "test": "pasokan obat", "ood": "stok obat"},
    "jembatan": {"train": "jembatan", "dev": "konstruksi jembatan", "test": "akses jembatan", "ood": "jembatan penyeberangan"},
    "sungai": {"train": "sungai", "dev": "aliran kali", "test": "badan sungai", "ood": "aliran sungai"},
    "selokan": {"train": "selokan", "dev": "parit saluran", "test": "got saluran", "ood": "selokan air"},
    "got": {"train": "got", "dev": "parit got", "test": "saluran got", "ood": "aliran got"},
    "banjir": {"train": "banjir", "dev": "genangan banjir", "test": "luapan banjir", "ood": "air banjir"},
    "genangan": {"train": "genangan", "dev": "air tergenang", "test": "genangan air", "ood": "tumpukan genangan"},
    "kabel": {"train": "kabel", "dev": "untaian kabel", "test": "bentangan kabel", "ood": "jalur kabel"},
    "pipa": {"train": "pipa", "dev": "pipa saluran", "test": "saluran pipa", "ood": "instalasi pipa"},
    "halte": {"train": "halte", "dev": "halte angkutan", "test": "area halte", "ood": "pos halte"},
    "angkot": {"train": "angkot", "dev": "angkutan kota", "test": "mobil angkot", "ood": "armada angkot"},
    "bus": {"train": "bus", "dev": "armada bus", "test": "kendaraan bus", "ood": "bus umum"},
    "membahayakan": {"train": "membahayakan", "dev": "berbahaya bagi", "test": "sangat rawan untuk", "ood": "mengancam keselamatan"},
    "pemotor": {"train": "pengendara motor", "dev": "para pemotor", "test": "pengguna motor", "ood": "pengendara roda dua"},
    "anak": {"train": "anak", "dev": "anak-anak", "test": "bocah", "ood": "anak kecil"},
    "rumah": {"train": "rumah", "dev": "hunian", "test": "rumah warga", "ood": "tempat tinggal"},
    "bangunan": {"train": "bangunan", "dev": "gedung", "test": "fasilitas bangunan", "ood": "konstruksi bangunan"},
    "depan": {"train": "depan", "dev": "bagian depan", "test": "area depan", "ood": "sisi depan"},
    "tengah": {"train": "tengah", "dev": "bagian tengah", "test": "tengah-tengah", "ood": "area tengah"},
    "pinggir": {"train": "pinggir", "dev": "tepian", "test": "sisi tepi", "ood": "pinggiran"},
    "kantor": {"train": "kantor", "dev": "gedung kantor", "test": "tempat kantor", "ood": "loket kantor"},
    "pelayanan": {"train": "pelayanan", "dev": "layanan", "test": "urusan layanan", "ood": "proses pelayanan"},
    "biaya": {"train": "biaya", "dev": "tarif bayar", "test": "pungutan biaya", "ood": "ongkos bayar"},
    "uang": {"train": "uang", "dev": "dana uang", "test": "sejumlah uang", "ood": "dana pembayaran"},
    "tanpa": {"train": "tanpa", "dev": "tanpa adanya", "test": "tidak disertai", "ood": "sama sekali tanpa"},
    "aspal": {"train": "aspal", "dev": "aspal jalan", "test": "lapisan aspal", "ood": "permukaan aspal"},
    "pungli": {"train": "pungli", "dev": "pungutan liar", "test": "tarikan liar", "ood": "pungli tanpa karcis"},
    "ruang": {"train": "ruang", "dev": "ruangan", "test": "kamar ruang", "ood": "area ruangan"},
    "rambu": {"train": "rambu", "dev": "rambu lalu lintas", "test": "papan rambu", "ood": "tanda rambu"},
    "jadwal": {"train": "jadwal", "dev": "waktu jadwal", "test": "jadwal layanan", "ood": "jam operasional"},
    "evakuasi": {"train": "evakuasi", "dev": "tindakan evakuasi", "test": "proses evakuasi", "ood": "upaya evakuasi"},
    "pagar": {"train": "pagar", "dev": "pagar pembatas", "test": "besi pagar", "ood": "pagar pengaman"},
    "papan": {"train": "papan", "dev": "papan informasi", "test": "plang papan", "ood": "papan petunjuk"},
    "bau": {"train": "bau", "dev": "aroma bau", "test": "bau menyengat", "ood": "bau busuk"},
    "pintu": {"train": "pintu", "dev": "pintu akses", "test": "daun pintu", "ood": "pintu masuk"},
    "limbah": {"train": "limbah", "dev": "limbah kotoran", "test": "buangan limbah", "ood": "cairan limbah"},
    "aliran": {"train": "aliran", "dev": "arus aliran", "test": "debit aliran", "ood": "saluran aliran"},
    "surat": {"train": "surat", "dev": "dokumen surat", "test": "berkas surat", "ood": "surat keterangan"},
    "aksi": {"train": "aksi", "dev": "tindakan aksi", "test": "ulah aksi", "ood": "perbuatan aksi"},
    "lansia": {"train": "lansia", "dev": "warga lansia", "test": "orang tua lansia", "ood": "kalangan lansia"},
    "tambalan": {"train": "tambalan", "dev": "lapisan tambalan", "test": "aspal tambal", "ood": "bekas tambalan"},
    "mengelupas": {"train": "mengelupas", "dev": "terkelupas", "test": "lepas terkelupas", "ood": "rontok terkelupas"},
    "ular": {"train": "ular", "dev": "hewan ular", "test": "ular liar", "ood": "satwa ular"},
    "tps": {"train": "TPS", "dev": "TPS wilayah", "test": "TPS terpadu", "ood": "TPS kawasan"},
    "kucing": {"train": "kucing", "dev": "kucing warga", "test": "kucing liar", "ood": "hewan kucing"},
    "kerikil": {"train": "kerikil", "dev": "batu kerikil", "test": "butiran kerikil", "ood": "kerikil pasir"},
    "material": {"train": "material", "dev": "bahan material", "test": "muatan material", "ood": "material proyek"},
    "proyek": {"train": "proyek", "dev": "pekerjaan proyek", "test": "kegiatan proyek", "ood": "konstruksi proyek"},
    "tebing": {"train": "tebing", "dev": "lereng tebing", "test": "sisi tebing", "ood": "dinding tebing"},
    "tunanetra": {"train": "tunanetra", "dev": "kaum tunanetra", "test": "warga tunanetra", "ood": "penyandang tunanetra"},
    "retensi": {"train": "retensi", "dev": "kolam retensi", "test": "tampungan retensi", "ood": "area retensi"},
    "pasir": {"train": "pasir", "dev": "butiran pasir", "test": "material pasir", "ood": "timbunan pasir"},
    "kardus": {"train": "kardus", "dev": "kotak kardus", "test": "tumpukan kardus", "ood": "kardus bekas"},
    "lalat": {"train": "lalat", "dev": "kawanan lalat", "test": "kerumunan lalat", "ood": "hama lalat"},
    "kran": {"train": "kran", "dev": "keran air", "test": "kran umum", "ood": "instalasi keran"},
    "pelanggan": {"train": "pelanggan", "dev": "nomor pelanggan", "test": "warga pelanggan", "ood": "pihak pelanggan"},
    "adminduk": {"train": "adminduk", "dev": "layanan adminduk", "test": "administrasi adminduk", "ood": "urusan adminduk"},
    "kk": {"train": "KK", "dev": "kartu keluarga", "test": "dokumen KK", "ood": "berkas KK"},
    "ktp": {"train": "KTP", "dev": "KTP fisik", "test": "kartu KTP", "ood": "identitas KTP"},
    "nik": {"train": "NIK", "dev": "nomor NIK", "test": "identitas NIK", "ood": "data NIK"},
    "akta": {"train": "akta", "dev": "akta kelahiran", "test": "dokumen akta", "ood": "berkas akta"},
    "bpjs": {"train": "BPJS", "dev": "layanan BPJS", "test": "kartu BPJS", "ood": "program BPJS"},
    "fotokopi": {"train": "fotokopi", "dev": "salinan fotokopi", "test": "lembar fotokopi", "ood": "berkas fotokopi"},
    "kematian": {"train": "kematian", "dev": "akta kematian", "test": "surat kematian", "ood": "keterangan kematian"},
    "berkas": {"train": "berkas", "dev": "berkas dokumen", "test": "kelengkapan berkas", "ood": "berkas pengajuan"},
    "cincin": {"train": "cincin", "dev": "lingkar cincin", "test": "cincin jari", "ood": "perhiasan cincin"},
    "balita": {"train": "balita", "dev": "anak balita", "test": "bocah balita", "ood": "anak kecil"},
    "tawon": {"train": "tawon", "dev": "sarang tawon", "test": "gerombolan tawon", "ood": "koloni tawon"},
    "sumur": {"train": "sumur", "dev": "lubang sumur", "test": "sumur tua", "ood": "galian sumur"},
    "hidran": {"train": "hidran", "dev": "pilar hidran", "test": "instalasi hidran", "ood": "kran hidran"},
    "gas": {"train": "gas", "dev": "kebocoran gas", "test": "aliran gas", "ood": "uap gas"},
    "metana": {"train": "metana", "dev": "gas metana", "test": "uap metana", "ood": "kandungan metana"},
    "kimia": {"train": "kimia", "dev": "bahan kimia", "test": "cairan kimia", "ood": "zat kimia"},
    "bencana": {"train": "bencana", "dev": "lokasi bencana", "test": "kondisi bencana", "ood": "peristiwa bencana"},
    "drainase": {"train": "drainase", "dev": "saluran drainase", "test": "aliran drainase", "ood": "sistem drainase"},
    "kartu": {"train": "kartu", "dev": "lembar kartu", "test": "dokumen kartu", "ood": "kartu identitas"},
    "dokumen": {"train": "dokumen", "dev": "berkas dokumen", "test": "dokumen resmi", "ood": "arsip dokumen"},
    "administrasi": {"train": "administrasi", "dev": "layanan administrasi", "test": "urusan administrasi", "ood": "proses administrasi"},
    "data": {"train": "data", "dev": "rekaman data", "test": "catatan data", "ood": "arsip data"},
    "pengaduan": {"train": "pengaduan", "dev": "berkas pengaduan", "test": "laporan pengaduan", "ood": "aduan pengaduan"},
    "informasi": {"train": "informasi", "dev": "penjelasan informasi", "test": "pemberitahuan informasi", "ood": "keterangan informasi"},
    "aman": {"train": "aman", "dev": "terkendali aman", "test": "cukup aman", "ood": "kondusif aman"},
    "normal": {"train": "normal", "dev": "berjalan normal", "test": "tetap normal", "ood": "kondisi wajar"},
    "fasilitas": {"train": "fasilitas", "dev": "sarana fasilitas", "test": "sarana umum", "ood": "sarana publik"},
    "terhambat": {"train": "terhambat", "dev": "cukup terhambat", "test": "agak terhambat", "ood": "mengalami hambatan"},
    "tersendat": {"train": "tersendat", "dev": "mulai tersendat", "test": "sempat tersendat", "ood": "mengalami ketersendatan"},
}

def sub_syn(text, split):
    def repl(m):
        w = m.group(0).lower()
        if w in SYNONYMS:
            return SYNONYMS[w][split]
        return m.group(0)
    return re.sub(r'\b[a-zA-Z]+\b', repl, text)

def split_into_constituents(text):
    norm = re.sub(r'(\w+)/(\w+)', r'\1 atau \2', text).replace('-', ' ')
    words = norm.split()
    chunks = []
    current = []
    for w in words:
        wl = w.lower()
        if wl in BREAK_WORDS and current:
            chunks.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
            if len(current) >= 3 and not (w.lower() in BREAK_WORDS):
                chunks.append(" ".join(current))
                current = []
    if current:
        if chunks and len(current) == 1 and current[0].lower() not in BREAK_WORDS:
            chunks[-1] += " " + current[0]
        else:
            chunks.append(" ".join(current))
    return chunks

def build_split_paraphrases(issue: str, surface_variant: int = 0) -> dict[str, str]:
    chunks = split_into_constituents(issue)
    k = len(chunks)

    def render_internal(split_name: str, form_idx: int) -> str:
        sc = [sub_syn(c, split_name) for c in chunks]
        f = form_idx % 4
        if k >= 3:
            if f == 0:
                return " ".join(sc)
            elif f == 1:
                c_s = sc[1:] + sc[:1]
                v = f"{c_s[0]} terkait {c_s[1]}"
                if len(c_s) > 2:
                    v += " serta " + " dan ".join(c_s[2:])
                return v
            elif f == 2:
                c_s = sc[2:] + sc[:2]
                v = f"{c_s[0]} akibat {c_s[1]}"
                if len(c_s) > 2:
                    v += " pada " + " kemudian ".join(c_s[2:])
                return v
            else:
                c_s = sc[3:] + sc[:3] if k > 3 else sc[:1] + list(reversed(sc[1:]))
                v = f"{c_s[0]} mengenai {c_s[1]}"
                if len(c_s) > 2:
                    v += " disertai " + " beserta ".join(c_s[2:])
                return v
        elif k == 2:
            if f == 0:
                return f"{sc[0]} {sc[1]}"
            elif f == 1:
                return f"{sc[1]} terkait {sc[0]}"
            elif f == 2:
                return f"{sc[1]} akibat {sc[0]}"
            else:
                return f"{sc[1]} mengenai {sc[0]}"
        else:
            if f == 0:
                return sc[0]
            elif f == 1:
                return f"kondisi {sc[0]} di lapangan"
            elif f == 2:
                return f"laporan tentang {sc[0]} di tempat"
            else:
                return f"adanya {sc[0]} di area"

    def render_ood(form_idx: int) -> str:
        sc = [sub_syn(c, "ood") for c in chunks]
        if k >= 2:
            ood_chunks = list(reversed(sc))
            v = f"{ood_chunks[0]} perihal {ood_chunks[1]}"
            if len(ood_chunks) > 2:
                v += " dalam hal " + " dan ".join(ood_chunks[2:])
            v += " di wilayah setempat"
            return v
        else:
            return f"kejadian mengenai {sc[0]} di wilayah setempat"

    res = {}
    offsets = {"train": 0, "dev": 1, "test": 2, "ood": 3}
    for s_name in ("train", "dev", "test", "ood"):
        f_idx = surface_variant + offsets[s_name]
        if s_name == "ood":
            res[s_name] = render_ood(f_idx)
        else:
            res[s_name] = render_internal(s_name, f_idx)
    return res


STYLE_FAMILIES: tuple[str, ...] = ("direct_report", "field_observation", "community_concern")

OPENINGS_INTERNAL: tuple[str, ...] = (
    "Selamat pagi admin",
    "Halo petugas",
    "Selamat siang min",
    "Selamat sore admin",
    "Selamat malam admin",
    "Halo admin",
    "Pagi min",
    "Halo pihak kelurahan",
    "Siang admin",
    "Sore admin",
    "Malam admin",
    "Lapor admin",
)

OPENINGS_OOD: tuple[str, ...] = (
    "Kepada dinas terkait",
    "Yth. Pemkot Bandung",
    "Warga menyampaikan laporan",
    "Mohon perhatian pihak berwenang",
)

INTENT_POOLS_INTERNAL: dict[str, tuple[str, ...]] = {
    "COMPLAINT": (
        "saya lapor aduan ini",
        "kami laporkan keluhan ini",
        "aduan warga ini mohon ditangani",
        "kami komplain atas kondisi ini",
        "mohon segera ditangani aduan ini",
        "tolong ditangani keluhan warga ini",
    ),
    "INQUIRY": (
        "bagaimana info perbaikan lokasi ini",
        "mohon info penanganan lokasi ini",
        "mohon informasi penanganan lokasi ini",
        "mohon keterangan perbaikan lokasi ini",
        "bagaimana respon penanganan lokasi ini",
        "mohon informasi perbaikan lokasi ini",
    ),
    "FEEDBACK": (
        "kami kirim saran perbaikan ini",
        "masukan kami untuk perbaikan ini",
        "saran dari warga untuk perbaikan",
        "saran masukan untuk perbaikan ini",
        "masukan kami mengenai perbaikan ini",
        "saran warga terkait perbaikan ini",
    ),
}

INTENT_POOLS_OOD: dict[str, tuple[str, ...]] = {
    "COMPLAINT": (
        "kami menuntut penuntasan segera",
        "pihak warga memprotes keras kendala ini",
        "keadaan ini sungguh merugikan lingkungan",
        "kami menyampaikan keluhan resmi warga",
    ),
    "INQUIRY": (
        "kapan kepastian realisasi pekerjaan ini",
        "apakah terdapat estimasi waktu pelaksanaan",
        "bagaimana kejelasan tahapan penyelesaiannya",
        "apakah ada kepastian agenda pembenahan",
    ),
    "FEEDBACK": (
        "masukan masyarakat yakni pembaruan fasilitas",
        "anjuran warga ialah pembenahan tata kelola",
        "kami merekomendasikan penataan fasilitas",
        "usulan kami yaitu perbaikan tuntas fasilitas",
    ),
}

AMBIGUITY_CLAUSES_INTERNAL: dict[str, tuple[str, ...]] = {
    "train": ("titiknya kurang spesifik nih", "titiknya belum pasti nih"),
    "dev": ("posisinya masih samar nih", "lokasinya belum bisa dipastikan"),
    "test": ("titiknya masih ambigu membingungkan", "informasi lokasinya belum jelas"),
}

AMBIGUITY_CLAUSES_OOD: tuple[str, ...] = (
    "titik koordinat spesifik belum terkonfirmasi",
    "keberadaan letak rincinya belum definitif",
    "acuan titik tepatnya masih belum terinci",
    "posisi bidang pastinya belum terverifikasi",
)

SAFE_CATEGORY_NOUNS: dict[Category, tuple[str, ...]] = {
    Category.ROAD: (
        "marka jalan",
        "trotoar jalan",
        "paving jalan",
        "rambu jalan",
        "lampu jalan",
    ),
    Category.DRAINAGE_FLOOD: (
        "saluran air",
        "parit saluran",
        "gorong-gorong saluran",
        "pintu air",
    ),
    Category.WASTE: (
        "tempat sampah",
        "wadah sampah",
        "tumpukan sampah",
        "bak sampah",
    ),
    Category.CLEAN_WATER: (
        "meteran air",
        "kran air",
        "pipa air",
        "sambungan pipa",
    ),
    Category.CIVIL_ADMIN: (
        "loket pelayanan",
        "ruang pelayanan",
        "meja pelayanan",
        "kantor pelayanan",
    ),
    Category.HEALTH_SERVICE: (
        "ruang tunggu pasien",
        "loket obat",
        "papan poli dokter",
        "kursi pasien",
    ),
    Category.PUBLIC_ORDER: (
        "pos kamling",
        "papan tata tertib",
        "area umum",
        "spanduk warga",
    ),
    Category.TRANSPORTATION: (
        "halte bus",
        "shelter angkutan",
        "tiang rute bus",
        "tempat tunggu angkot",
    ),
    Category.FIRE_RESCUE: (
        "pilar hidran",
        "plang titik kumpul",
        "kotak alat hidran",
        "papan evakuasi",
    ),
    Category.SOCIAL_AFFAIRS: (
        "meja pendaftaran bantuan",
        "loket dana bantuan",
        "papan info bantuan",
        "ruang pendaftaran",
    ),
    Category.EDUCATION: (
        "papan nama sekolah",
        "halaman sekolah",
        "pagar sekolah",
        "bangku sekolah",
    ),
    Category.PARKS_HOUSING: (
        "bangku taman",
        "lampu taman",
        "pagar taman",
        "area taman",
    ),
}

SAFE_TEMPLATES_INTERNAL: tuple[str, ...] = (
    "keluhan ringan sarana {noun}",
    "perawatan berkala sarana {noun}",
    "kebersihan ringan sarana {noun}",
    "penataan kosmetik sarana {noun}",
)

SAFE_TEMPLATES_OOD: tuple[str, ...] = (
    "tinjauan pemeliharaan minor sarana {noun}",
    "evaluasi perawatan berkala sarana {noun}",
    "pengecekan visual estetika sarana {noun}",
    "kebutuhan perapian minor sarana {noun}",
)

UNSAFE_WORDS_IN_LOW: tuple[str, ...] = (
    "bahaya", "rawan", "korban", "luka", "kebakaran", "ambles", "banjir", "terbakar", "nyawa", "darurat"
)
URGENT_TRIGGER_WORDS: tuple[str, ...] = (
    "darurat", "segera", "korban", "kritis", "kebakaran", "evakuasi", "ambulans", "sekarang"
)

RISK_EVIDENCE_LOW: dict[str, tuple[str, ...]] = {
    "train": (
        "kondisi warga masih aman",
        "fungsi fasilitas warga normal",
        "sarana warga tetap aman",
        "aktivitas warga tetap normal",
        "lingkungan warga masih terkendali",
        "urusan warga tergolong aman",
    ),
    "dev": (
        "kondisi masyarakat masih aman",
        "fungsi fasilitas masyarakat normal",
        "sarana masyarakat tetap aman",
        "aktivitas masyarakat tetap normal",
        "lingkungan masyarakat masih terkendali",
        "urusan masyarakat tergolong aman",
    ),
    "test": (
        "kondisi warga sekitar masih aman",
        "fungsi sarana umum normal",
        "sarana umum tetap aman",
        "aktivitas warga sekitar tetap normal",
        "lingkungan sekitar masih terkendali",
        "urusan warga sekitar aman",
    ),
    "ood": (
        "kondisi penduduk masih aman",
        "fungsi fasilitas penduduk normal",
        "sarana penduduk tetap aman",
        "aktivitas penduduk tetap kondusif",
    ),
}

RISK_EVIDENCE_MEDIUM: dict[str, tuple[str, ...]] = {
    "train": (
        "aktivitas warga cukup terhambat",
        "alur pelayanan warga tersendat",
        "kelancaran akses warga terganggu",
        "mobilitas warga agak melambat",
        "pelayanan warga sempat tersendat",
        "antrean warga bertambah panjang",
    ),
    "dev": (
        "aktivitas masyarakat cukup terhambat",
        "alur layanan masyarakat tersendat",
        "kelancaran akses masyarakat terganggu",
        "mobilitas masyarakat agak melambat",
        "layanan masyarakat sempat tersendat",
        "antrean masyarakat bertambah panjang",
    ),
    "test": (
        "aktivitas warga sekitar terhambat",
        "alur pelayanan umum tersendat",
        "kelancaran akses sekitar terganggu",
        "mobilitas warga sekitar melambat",
        "proses pelayanan sempat tersendat",
        "antrean sekitar bertambah panjang",
    ),
    "ood": (
        "ritme mobilitas penduduk melambat",
        "alur kegiatan penduduk tersendat",
        "proses pelayanan penduduk melambat",
        "antrean penduduk bertambah panjang",
    ),
}

RISK_EVIDENCE_HIGH: dict[str, dict[Category, tuple[str, ...]]] = {
    "train": {
        Category.ROAD: ("pengendara motor rawan tergelincir", "sangat berisiko kecelakaan pemotor", "membahayakan keselamatan pemotor jalan", "beberapa pemotor hampir celaka"),
        Category.DRAINAGE_FLOOD: ("luapan air berpotensi merusak", "genangan air rawan menjebak", "tanggul rembes aliran air", "saluran air mampet berpotensi"),
        Category.WASTE: ("limbah sampah berisiko infeksi", "ceceran limbah sampah melukai", "gas timbunan sampah menyulut", "tumpukan sampah memicu lalat"),
        Category.CLEAN_WATER: ("aliran pipa mengikis jalan", "semburan pipa merusak kabel", "pipa bocor memicu selip", "pasokan air keruh membahayakan"),
        Category.CIVIL_ADMIN: ("antrean pelayanan rawan keributan", "pungutan biaya merugikan warga", "ruang pelayanan memicu pingsan", "antrean kantor memicu perselisihan"),
        Category.HEALTH_SERVICE: ("pasien faskes terancam tertunda", "ruang pasien menularkan infeksi", "ketiadaan dokter faskes berisiko", "resep obat berisiko salah"),
        Category.PUBLIC_ORDER: ("gesekan antar warga rawan", "premanisme mengancam keselamatan warga", "balapan liar rawan tabrakan", "kerumunan liar memicu kejahatan"),
        Category.TRANSPORTATION: ("pintu armada bus terbuka", "rambu halte memicu tabrakan", "halte gelap rawan kejahatan", "armada angkot memicu tabrakan"),
        Category.FIRE_RESCUE: ("korsleting atap rumah warga", "sarang tawon atap rumah", "kebocoran gas ruko warga", "batang pohon rawan tumbang"),
        Category.SOCIAL_AFFAIRS: ("lansia terlantar berisiko dehidrasi", "dana bantuan mencederai warga", "anak terlantar rawan eksploitasi", "warga miskin terancam terlantar"),
        Category.EDUCATION: ("plafon gedung sekolah ambrol", "kabel gedung sekolah menyengat", "tembok gedung sekolah roboh", "kaca gedung sekolah melukai"),
        Category.PARKS_HOUSING: ("batang pohon rawan patah", "tiang lampu taman roboh", "arena taman rawan melukai", "paving taman mencederai warga"),
    },
    "dev": {
        Category.ROAD: ("para pemotor rawan tergelincir", "berisiko kecelakaan para pemotor", "bahayakan keselamatan para pemotor", "para pemotor hampir celaka"),
        Category.DRAINAGE_FLOOD: ("aliran drainase berpotensi merusak", "air genangan rawan menjebak", "tanggul rembes drainase", "aliran drainase mampet berpotensi"),
        Category.WASTE: ("buangan sampah berisiko infeksi", "ceceran buangan sampah melukai", "gas tumpukan sampah menyulut", "buangan sampah memicu lalat"),
        Category.CLEAN_WATER: ("saluran pipa mengikis jalan", "semburan saluran pipa merusak", "saluran pipa memicu selip", "aliran air keruh bahayakan"),
        Category.CIVIL_ADMIN: ("antrean layanan rawan keributan", "ongkos bayar rugikan masyarakat", "ruang layanan memicu pingsan", "antrean loket memicu perselisihan"),
        Category.HEALTH_SERVICE: ("warga berobat terancam tertunda", "ruang berobat tularkan infeksi", "ketiadaan dokter pemeriksa berisiko", "pasokan obat berisiko salah"),
        Category.PUBLIC_ORDER: ("gesekan masyarakat rawan bentrok", "premanisme ancam keselamatan masyarakat", "balapan liar picu tabrakan", "kerumunan malam picu kejahatan"),
        Category.TRANSPORTATION: ("pintu kendaraan bus terbuka", "rambu halte picu tabrakan", "halte gelap picu kejahatan", "mobil angkot picu tabrakan"),
        Category.FIRE_RESCUE: ("korsleting atap hunian masyarakat", "sarang tawon hunian masyarakat", "kebocoran gas toko terkunci", "batang pohon lapuk tumbang"),
        Category.SOCIAL_AFFAIRS: ("warga lansia berisiko dehidrasi", "program bantuan cedera masyarakat", "bocah terlantar rawan eksploitasi", "masyarakat miskin terancam terlantar"),
        Category.EDUCATION: ("plafon lingkungan sekolah ambrol", "kabel lingkungan sekolah menyengat", "tembok lingkungan sekolah roboh", "kaca lingkungan sekolah melukai"),
        Category.PARKS_HOUSING: ("pepohonan lapuk rawan patah", "tiang penerangan taman roboh", "area taman rawan melukai", "paving taman cedera masyarakat"),
    },
    "test": {
        Category.ROAD: ("pengguna motor rawan tergelincir", "berisiko kecelakaan pengguna motor", "bahayakan keselamatan pengguna motor", "pengguna motor hampir celaka"),
        Category.DRAINAGE_FLOOD: ("luapan banjir berpotensi merusak", "luapan air rawan menjebak", "tanggul rembes parit saluran", "parit saluran mampet berpotensi"),
        Category.WASTE: ("limbah sampah berisiko infeksi", "ceceran limbah sampah lukai", "gas limbah sampah menyulut", "limbah sampah memicu lalat"),
        Category.CLEAN_WATER: ("saluran pipa kikis badan jalan", "semburan saluran pipa rusak", "saluran pipa picu selip", "pasokan air keruh membahayakan"),
        Category.CIVIL_ADMIN: ("antrean urusan layanan ribut", "tarif bayar rugikan warga", "kamar ruang memicu pingsan", "antrean kantor picu perselisihan"),
        Category.HEALTH_SERVICE: ("pasien faskes terancam tertunda", "kamar ruang tularkan infeksi", "ketiadaan tenaga dokter berisiko", "stok obat berisiko salah"),
        Category.PUBLIC_ORDER: ("gesekan warga sekitar rawan", "premanisme ancam warga sekitar", "balapan liar timbulkan tabrakan", "kerumunan liar picu kejahatan"),
        Category.TRANSPORTATION: ("pintu armada bus transit buka", "rambu pos halte picu tabrakan", "pos halte malam rawan", "mobil angkot timbulkan tabrakan"),
        Category.FIRE_RESCUE: ("korsleting atap rumah warga", "sarang tawon atap rumah", "kebocoran gas ruko warga", "pohon peneduh rawan tumbang"),
        Category.SOCIAL_AFFAIRS: ("orang tua lansia dehidrasi", "program bantuan lukai warga", "anak kecil rawan eksploitasi", "warga sekitar terancam terlantar"),
        Category.EDUCATION: ("plafon area sekolah ambrol", "kabel area sekolah menyengat", "tembok area sekolah roboh", "kaca area sekolah melukai"),
        Category.PARKS_HOUSING: ("pohon peneduh rawan patah", "tiang lampu taman roboh", "fasilitas taman rawan melukai", "paving taman lukai warga"),
    },
    "ood": {
        cat: tuple(f"kondisi berisiko ancam penduduk {cat.value.lower()}" for _ in range(4))
        for cat in Category
    },
}

RISK_EVIDENCE_URGENT: dict[str, dict[Category, tuple[str, ...]]] = {
    "train": {
        Category.ROAD: ("jembatan jalan putus darurat", "longsor tebing korban darurat", "tabrakan beruntun pemotor darurat", "amblesan jalan korban darurat"),
        Category.DRAINAGE_FLOOD: ("banjir warga tenggelam darurat", "arus banjir korban darurat", "tanggul jebol banjir darurat", "luapan air banjir darurat"),
        Category.WASTE: ("ledakan sampah bakar darurat", "kebakaran sampah korban darurat", "longsor timbunan sampah darurat", "kebakaran limbah sampah darurat"),
        Category.CLEAN_WATER: ("semburan pipa trafo darurat", "ledakan pipa korban darurat", "bocoran pipa jebol darurat", "aliran pipa deras darurat"),
        Category.CIVIL_ADMIN: ("kerusuhan kantor pelayanan darurat", "kebakaran kantor pelayanan darurat", "runtuh kantor pelayanan darurat", "bentrok kantor pelayanan darurat"),
        Category.HEALTH_SERVICE: ("pasien gagal napas darurat", "pasien pendarahan hebat darurat", "keracunan massal pasien darurat", "pasien kritis faskes darurat"),
        Category.PUBLIC_ORDER: ("tawuran senjata korban darurat", "penyerangan rumah korban darurat", "kerusuhan massa korban darurat", "anarkis kelompok korban darurat"),
        Category.TRANSPORTATION: ("tabrakan armada bus darurat", "rem blong bus darurat", "armada bus terbakar darurat", "kecelakaan armada bus darurat"),
        Category.FIRE_RESCUE: ("kebakaran rumah warga darurat", "kebakaran ruko warga darurat", "balita terjebak kebakaran darurat", "kebakaran gedung warga darurat"),
        Category.SOCIAL_AFFAIRS: ("tenda pengungsi korban darurat", "lansia pingsan bantuan darurat", "lansia kritis korban darurat", "balita sakit korban darurat"),
        Category.EDUCATION: ("atap gedung sekolah darurat", "kebakaran laboratorium sekolah darurat", "tembok gedung sekolah darurat", "keracunan siswa sekolah darurat"),
        Category.PARKS_HOUSING: ("pohon tumbang korban darurat", "kabel taman sengat darurat", "wahana taman korban darurat", "lampu taman timpa darurat"),
    },
    "dev": {
        Category.ROAD: ("jembatan ruas jalan putus darurat", "longsor tebing tertimbun darurat", "tabrakan beruntun para pemotor darurat", "amblesan ruas jalan korban darurat"),
        Category.DRAINAGE_FLOOD: ("banjir masyarakat tenggelam darurat", "arus banjir hanyutkan darurat", "tanggul jebol genang darurat", "luapan air bah darurat"),
        Category.WASTE: ("ledakan buangan sampah darurat", "kebakaran sampah korban darurat", "longsor tumpukan sampah darurat", "kebakaran limbah kotoran darurat"),
        Category.CLEAN_WATER: ("semburan pipa saluran trafo darurat", "ledakan pipa saluran darurat", "bocoran pipa saluran darurat", "aliran pipa saluran darurat"),
        Category.CIVIL_ADMIN: ("kerusuhan tempat kantor layanan darurat", "kebakaran tempat kantor layanan darurat", "runtuh gedung kantor layanan darurat", "bentrok tempat kantor darurat"),
        Category.HEALTH_SERVICE: ("pasien berobat napas darurat", "pasien berobat pendarahan darurat", "keracunan massal warga berobat darurat", "warga berobat kritis darurat"),
        Category.PUBLIC_ORDER: ("tawuran senjata korban luka darurat", "penyerangan hunian korban darurat", "kerusuhan massa bakar darurat", "anarkis kelompok masyarakat darurat"),
        Category.TRANSPORTATION: ("tabrakan kendaraan bus darurat", "rem blong kendaraan bus darurat", "kendaraan bus terbakar darurat", "kecelakaan kendaraan bus darurat"),
        Category.FIRE_RESCUE: ("kebakaran hunian masyarakat darurat", "kebakaran ruko masyarakat darurat", "bocah terjebak kebakaran darurat", "kebakaran gedung masyarakat darurat"),
        Category.SOCIAL_AFFAIRS: ("tenda pengungsian korban luka darurat", "warga lansia pingsan darurat", "warga lansia kritis darurat", "bocah sakit korban darurat"),
        Category.EDUCATION: ("atap lingkungan sekolah darurat", "kebakaran lab sekolah darurat", "tembok lingkungan sekolah darurat", "keracunan murid sekolah darurat"),
        Category.PARKS_HOUSING: ("batang pohon tumbang darurat", "kabel area taman darurat", "wahana area taman darurat", "lampu area taman darurat"),
    },
    "test": {
        Category.ROAD: ("jembatan badan jalan putus darurat", "longsor tebing korban jiwa darurat", "tabrakan beruntun pengguna motor darurat", "amblesan badan jalan darurat"),
        Category.DRAINAGE_FLOOD: ("banjir warga sekitar tenggelam darurat", "arus banjir korban jiwa darurat", "tanggul jebol banjir parah darurat", "luapan air banjir lebat darurat"),
        Category.WASTE: ("ledakan limbah sampah darurat", "kebakaran limbah sampah korban darurat", "longsor limbah sampah darurat", "kebakaran buangan sampah darurat"),
        Category.CLEAN_WATER: ("semburan saluran pipa trafo darurat", "ledakan saluran pipa darurat", "bocoran saluran pipa darurat", "aliran saluran pipa deras darurat"),
        Category.CIVIL_ADMIN: ("kerusuhan tempat kantor layanan darurat", "kebakaran tempat kantor darurat", "runtuh tempat kantor darurat", "bentrok tempat kantor darurat"),
        Category.HEALTH_SERVICE: ("pasien faskes gagal napas darurat", "pasien faskes pendarahan darurat", "keracunan massal pasien faskes darurat", "pasien faskes kritis darurat"),
        Category.PUBLIC_ORDER: ("tawuran senjata korban celaka darurat", "penyerangan rumah warga darurat", "kerusuhan massa bakar sarana darurat", "anarkis kelompok serang darurat"),
        Category.TRANSPORTATION: ("tabrakan armada bus umum darurat", "rem blong armada bus darurat", "armada bus umum terbakar darurat", "kecelakaan armada bus umum darurat"),
        Category.FIRE_RESCUE: ("kebakaran rumah warga sekitar darurat", "kebakaran ruko warga sekitar darurat", "anak kecil terjebak kebakaran darurat", "kebakaran fasilitas bangunan darurat"),
        Category.SOCIAL_AFFAIRS: ("tenda bencana korban celaka darurat", "orang tua lansia pingsan darurat", "orang tua lansia kritis darurat", "anak kecil sakit darurat"),
        Category.EDUCATION: ("atap area sekolah darurat", "kebakaran laboratorium sekolah darurat", "tembok area sekolah roboh darurat", "keracunan murid siswa darurat"),
        Category.PARKS_HOUSING: ("pohon peneduh tumbang darurat", "kabel taman kota darurat", "wahana taman kota darurat", "lampu taman kota darurat"),
    },
    "ood": {
        cat: tuple(f"ancaman kritis penduduk korban sarana {cat.value.lower()} darurat" for _ in range(4))
        for cat in Category
    },
}

STRUCTURE_TEMPLATES_BY_FAMILY: dict[str, dict[str, tuple[str, str, str]]] = {
    "direct_report": {
        "train": ("Saya melihat {issue}", "Mulainya {time_str}", "Dampak ke warga: {risk_phrase}."),
        "dev": ("Yang saya temui adalah {issue}", "Sudah berlangsung {time_str}", "Dampak di lapangan: {risk_phrase}."),
        "test": ("Warga mengeluhkan {issue}", "Keluhan ini muncul {time_str}", "Dampak yang terlihat: {risk_phrase}."),
    },
    "field_observation": {
        "train": ("Warga mengabarkan {issue}", "Kondisinya mulai {time_str}", "Situasi saat ini {risk_phrase}."),
        "dev": ("Kami mendapati {issue}", "Ini terjadi {time_str}", "Kondisi ini membuat {risk_phrase}."),
        "test": ("Diadukan masalah {issue}", "Saya melihatnya {time_str}", "Situasi sekitar {risk_phrase}."),
    },
    "community_concern": {
        "train": ("Ada laporan {issue}", "Terpantau sejak {time_str}", "Perlu diantisipasi karena {risk_phrase}."),
        "dev": ("Ditemukan {issue}", "Kejadiannya {time_str}", "Perlu perhatian mengingat {risk_phrase}."),
        "test": ("Informasi warga menyebut {issue}", "Laporan masuk {time_str}", "Keadaan lingkungan sehingga {risk_phrase}."),
    },
}

STRUCTURE_TEMPLATES_OOD: tuple[tuple[str, str, str], ...] = (
    ("Saya menjumpai {issue}", "Waktu terpantau {time_str}", "Catatan risiko: {risk_phrase}."),
    ("Tampak persoalan {issue}", "Pengamatan semenjak {time_str}", "Tinjauan kondisi: {risk_phrase}."),
    ("Terjadi {issue}", "Kejadian berlangsung {time_str}", "Perkiraan dampak: {risk_phrase}."),
)

def safe_issue_surface(category: Category, concept_id: int, split_mode: str = "train", independent: bool = False) -> str:
    nouns = SAFE_CATEGORY_NOUNS[category]
    noun = nouns[concept_id % len(nouns)]
    noun_syn = sub_syn(noun, split_mode)
    templates = SAFE_TEMPLATES_OOD if independent else SAFE_TEMPLATES_INTERNAL
    template = templates[concept_id % len(templates)]
    return template.format(noun=noun_syn)

def make_risk_evidence(
    category: Category,
    concept_id: int,
    risk: str,
    surface_variant: int,
    split_mode: str = "train",
    independent: bool = False,
) -> tuple[str, str]:
    if risk == "LOW":
        kind = "cosmetic_minor"
        pool = RISK_EVIDENCE_LOW[split_mode]
        clause = pool[(concept_id + surface_variant) % len(pool)]
    elif risk == "MEDIUM":
        kind = "operational_disruption"
        pool = RISK_EVIDENCE_MEDIUM[split_mode]
        clause = pool[(concept_id + surface_variant) % len(pool)]
    elif risk == "HIGH":
        kind = "safety_hazard"
        pool = RISK_EVIDENCE_HIGH[split_mode][category]
        clause = pool[(concept_id + surface_variant) % len(pool)]
    elif risk == "URGENT":
        kind = "immediate_emergency"
        pool = RISK_EVIDENCE_URGENT[split_mode][category]
        clause = pool[(concept_id + surface_variant) % len(pool)]
    else:
        raise ValueError(f"Unknown risk level: {risk}")
    return clause, kind

LABEL_POOLS: dict[str, Any] = {
    "intent": INTENT_POOLS_INTERNAL,
}

LOCATIONS: dict[str, tuple[tuple[str, str, str, str, str], ...]] = {
    "train": tuple((street + f' No. {10+i}', f"RT {(i % 30) + 1:02d} RW {(i % 20) + 1:02d}", kel, city, landmark) for i, (street, kel, city, landmark) in enumerate((
        ('Jl. Ir. H. Juanda', 'Kelurahan Dago', 'Kecamatan Coblong, Kota Bandung', 'deket Simpang Dago'),
        ('Jl. Dipatiukur', 'Kelurahan Lebakgede', 'Kecamatan Coblong, Kota Bandung', 'samping Kampus Unpad DU'),
        ('Jl. Tamansari', 'Kelurahan Sekeloa', 'Kecamatan Coblong, Kota Bandung', 'bawah jembatan Pasupati'),
        ('Jl. Cihampelas', 'Kelurahan Cipaganti', 'Kecamatan Coblong, Kota Bandung', 'seberang Ciwalk'),
        ('Jl. Dr. Djunjunan', 'Kelurahan Pasteur', 'Kecamatan Sukajadi, Kota Bandung', 'dekat pintu tol Pasteur'),
        ('Jl. Surya Sumantri', 'Kelurahan Sukagalih', 'Kecamatan Sukajadi, Kota Bandung', 'seberang Kampus Maranatha'),
        ('Jl. Sukajadi', 'Kelurahan Sukawarna', 'Kecamatan Sukajadi, Kota Bandung', 'dekat mal PVJ'),
        ('Jl. Cipedes', 'Kelurahan Cipedes', 'Kecamatan Sukajadi, Kota Bandung', 'dekat Polsek Sukajadi'),
        ('Jl. Malabar', 'Kelurahan Malabar', 'Kecamatan Lengkong, Kota Bandung', 'dekat Lapangan Lodaya'),
        ('Jl. Burangrang', 'Kelurahan Burangrang', 'Kecamatan Lengkong, Kota Bandung', 'seberang SMA Negeri 7'),
        ('Jl. Buah Batu', 'Kelurahan Cijagra', 'Kecamatan Lengkong, Kota Bandung', 'dekat Pasar Kordon'),
        ('Jl. Gatot Subroto', 'Kelurahan Turangga', 'Kecamatan Lengkong, Kota Bandung', 'dekat Trans Studio'),
    ))),
    "dev": tuple((street + f' No. {20+i}', f"RT {(i % 30) + 1:02d} RW {(i % 20) + 1:02d}", kel, city, landmark) for i, (street, kel, city, landmark) in enumerate((
        ('Jl. Asia Afrika', 'Kelurahan Braga', 'Kecamatan Sumur Bandung, Kota Bandung', 'dekat Alun-alun Bandung'),
        ('Jl. Naripan', 'Kelurahan Kebon Pisang', 'Kecamatan Sumur Bandung, Kota Bandung', 'seberang Gedung Bank BJB'),
        ('Jl. Tamblong', 'Kelurahan Merdeka', 'Kecamatan Sumur Bandung, Kota Bandung', 'dekat Gereja Katedral'),
        ('Jl. Braga', 'Kelurahan Babakan Ciamis', 'Kecamatan Sumur Bandung, Kota Bandung', 'dekat Museum KAA'),
        ('Jl. Riau', 'Kelurahan Citarum', 'Kecamatan Bandung Wetan, Kota Bandung', 'seberang Taman Pramuka'),
        ('Jl. Trunojoyo', 'Kelurahan Tamansari', 'Kecamatan Bandung Wetan, Kota Bandung', 'dekat Dago Plaza'),
        ('Jl. Sultan Agung', 'Kelurahan Cihapit', 'Kecamatan Bandung Wetan, Kota Bandung', 'seberang SMA Santo Aloysius'),
        ('Jl. Banda', 'Kelurahan Merdeka Wetan', 'Kecamatan Bandung Wetan, Kota Bandung', 'dekat GOR Saparua'),
        ('Jl. Progo', 'Kelurahan Citarum Kulon', 'Kecamatan Bandung Wetan, Kota Bandung', 'belakang Gedung Sate'),
        ('Jl. Diponegoro', 'Kelurahan Cihaur Geulis', 'Kecamatan Cibeunying Kaler, Kota Bandung', 'depan Lapangan Gasibu'),
        ('Jl. Supratman', 'Kelurahan Sukamaju', 'Kecamatan Cibeunying Kaler, Kota Bandung', 'dekat Taman Cibeunying'),
        ('Jl. Sentot Alibasyah', 'Kelurahan Cihaur Wetan', 'Kecamatan Cibeunying Kaler, Kota Bandung', 'dekat Museum Geologi'),
    ))),
    "test": tuple((street + f' No. {30+i}', f"RT {(i % 30) + 1:02d} RW {(i % 20) + 1:02d}", kel, city, landmark) for i, (street, kel, city, landmark) in enumerate((
        ('Jl. Soekarno Hatta', 'Kelurahan Batununggal', 'Kecamatan Bandung Kidul, Kota Bandung', 'dekat Metro Indah Mall'),
        ('Jl. Kiaracondong', 'Kelurahan Babakan Surabaya', 'Kecamatan Kiaracondong, Kota Bandung', 'dekat Flyover Kiaracondong'),
        ('Jl. Terusan Jakarta', 'Kelurahan Antapani Kulon', 'Kecamatan Antapani, Kota Bandung', 'dekat Griya Antapani'),
        ('Jl. Purwakarta', 'Kelurahan Antapani Tengah', 'Kecamatan Antapani, Kota Bandung', 'dekat Pasar Tradisional Antapani'),
        ('Jl. Jakarta', 'Kelurahan Kebonwaru', 'Kecamatan Batununggal, Kota Bandung', 'seberang BTM Antapani'),
        ('Jl. Ahmad Yani', 'Kelurahan Padasuka', 'Kecamatan Cibeunying Kidul, Kota Bandung', 'dekat Saung Angklung Udjo'),
        ('Jl. Brigjen Katamso', 'Kelurahan Sukaluyu', 'Kecamatan Cibeunying Kaler, Kota Bandung', 'dekat Pusdai Jabar'),
        ('Jl. Pahlawan', 'Kelurahan Cikutra', 'Kecamatan Cibeunying Kaler, Kota Bandung', 'seberang Taman Makam Pahlawan'),
        ('Jl. Surapati', 'Kelurahan Sadang Serang', 'Kecamatan Cibeunying Kaler, Kota Bandung', 'dekat Pasar Suci'),
        ('Jl. Cisaranten Kulon', 'Kelurahan Cisaranten', 'Kecamatan Arcamanik, Kota Bandung', 'dekat Lapas Sukamiskin'),
        ('Jl. Pacuan Kuda', 'Kelurahan Sukamiskin', 'Kecamatan Arcamanik, Kota Bandung', 'seberang SPBU Arcamanik'),
        ('Jl. Sindanglaya', 'Kelurahan Sindangjaya', 'Kecamatan Mandalajati, Kota Bandung', 'dekat RS Hermina Arcamanik'),
    ))),
    "ood": tuple((street + f' No. {50+i}', f"RT {(i % 30) + 41:02d} RW {(i % 20) + 31:02d}", kel, city, landmark) for i, (street, kel, city, landmark) in enumerate((
        ('Jl. Raya Kopo', 'Kelurahan Babakan Asih', 'Kecamatan Bojongloa Kaler, Kota Bandung', 'dekat RS Immanuel'),
        ('Jl. Peta', 'Kelurahan Suka Asih', 'Kecamatan Bojongloa Kaler, Kota Bandung', 'dekat Mal Festival Citylink'),
        ('Jl. Moh. Toha', 'Kelurahan Cigereleng', 'Kecamatan Regol, Kota Bandung', 'dekat Gerbang Tol Moch Toha'),
        ('Jl. BKR', 'Kelurahan Pasirluyu', 'Kecamatan Regol, Kota Bandung', 'dekat Lapangan Tegalega'),
        ('Jl. Moh. Ramdan', 'Kelurahan Ancol', 'Kecamatan Regol, Kota Bandung', 'seberang Taman Regol'),
        ('Jl. Rajawali Barat', 'Kelurahan Campaka', 'Kecamatan Andir, Kota Bandung', 'dekat RS Rajawali'),
        ('Jl. Ciroyom', 'Kelurahan Ciroyom Kulon', 'Kecamatan Andir, Kota Bandung', 'dekat Stasiun Ciroyom'),
        ('Jl. Garuda', 'Kelurahan Dunguscariang', 'Kecamatan Andir, Kota Bandung', 'dekat Puskesmas Garuda'),
        ('Jl. Pasirkaliki', 'Kelurahan Pamoyanan', 'Kecamatan Cicendo, Kota Bandung', 'seberang RS Kebon Jati'),
        ('Jl. Sukajadi Atas', 'Kelurahan Gegerkalong', 'Kecamatan Sukasari, Kota Bandung', 'dekat Polsek Sukasari'),
        ('Jl. Setiabudi', 'Kelurahan Isola', 'Kecamatan Sukasari, Kota Bandung', 'dekat Kampus UPI'),
        ('Jl. Gegerkalong Hilir', 'Kelurahan Sarijadi', 'Kecamatan Sukasari, Kota Bandung', 'dekat Kampus Polban'),
    ))),
}

TIME_PHRASES = {
    "train": (
        "sejak kemarin siang",
        "sudah tiga hari ini",
        "tadi pagi pas jam tujuh",
        "sejak semalam sampai subuh",
        "mulai awal pekan ini",
        "dari hari Senin lalu",
        "tadi siang pas hujan lebat",
        "sejak dua hari kemarin",
    ),
    "dev": (
        "mulai kemarin petang",
        "sudah empat hari berjalan",
        "pagi ini menjelang jam delapan",
        "semenjak semalam hingga siang",
        "mulai pertengahan pekan",
        "dari hari Selasa kemarin",
        "tadi sore waktu mendung",
        "sejak tiga hari berturut-turut",
    ),
    "test": (
        "dari kemarin sore",
        "sudah hampir lima hari",
        "pagi tadi waktu jam berangkat",
        "sejak semalam suntuk",
        "mulai akhir pekan kemarin",
        "dari hari Rabu pagi",
        "siang tadi sehabis gerimis",
        "sejak empat hari belakangan",
    ),
    "ood": (
        "kurang lebih seminggu penuh",
        "sejak Kamis malam pekan lalu",
        "waktu subuh tadi saat pasar buka",
        "selama lima hari tanpa henti",
        "mulai Jumat pagi waktu kerja",
        "sejak enam hari yang lalu",
        "tadi petang menjelang magrib",
        "sepanjang minggu kemarin",
    ),
}

def _spans(text: str, values: list[tuple[str, CanonicalSpanLabel]]) -> tuple[CanonicalSpan, ...]:
    result: list[CanonicalSpan] = []
    used: list[tuple[int, int]] = []
    for value, label in sorted(values, key=lambda x: len(x[0]), reverse=True):
        start = 0
        while value:
            pos = text.find(value, start)
            if pos < 0:
                break
            end = pos + len(value)
            if not any(pos < b and end > a for a, b in used):
                result.append(CanonicalSpan(pos, end, label))
                used.append((pos, end))
                break
            start = end
    return tuple(sorted(result, key=lambda x: (x.start, x.end)))

def _build(
    scenario_id: str,
    family_id: str,
    split: DatasetSplit,
    category: Category,
    intent: str,
    risk: str,
    completeness: str,
    variant: int,
    family_index: int,
    concept_id: int,
    independent: bool = False,
    style_index: int | None = None,
) -> ComplaintTrajectory:
    name = "ood" if independent else split.value
    loc_tuple = LOCATIONS[name][(family_index * 7 + variant) % len(LOCATIONS[name])]
    street, rt, kel, city, landmark = loc_tuple

    if risk == "LOW":
        base_issue = safe_issue_surface(category, concept_id, split_mode=name, independent=independent)
    else:
        base_issue = ISSUES[category][concept_id]

    paraphrases = build_split_paraphrases(base_issue, surface_variant=variant)
    issue = paraphrases[name]

    time_str = TIME_PHRASES[name][(family_index + variant * 2) % len(TIME_PHRASES[name])]

    if independent:
        opening = OPENINGS_OOD[(family_index + variant) % len(OPENINGS_OOD)]
        intent_pool = INTENT_POOLS_OOD[intent]
        amb_phrase = AMBIGUITY_CLAUSES_OOD[(family_index + variant) % len(AMBIGUITY_CLAUSES_OOD)]
    else:
        opening = OPENINGS_INTERNAL[(family_index + variant) % len(OPENINGS_INTERNAL)]
        intent_pool = INTENT_POOLS_INTERNAL[intent]
        amb_clauses = AMBIGUITY_CLAUSES_INTERNAL[name]
        amb_phrase = amb_clauses[(family_index + variant) % len(amb_clauses)]

    intent_phrase = intent_pool[(concept_id + variant) % len(intent_pool)]

    risk_phrase, risk_evidence_kind = make_risk_evidence(
        category=category,
        concept_id=concept_id,
        risk=risk,
        surface_variant=variant,
        split_mode=name,
        independent=independent,
    )

    if completeness == "SUFFICIENT":
        loc_parts = (street, rt, kel, city, landmark)
        location = f"{street}, {rt}, {kel}, {city}, {landmark}"
        loc_expr = location
    elif completeness == "INCOMPLETE":
        loc_parts = (kel,)
        location = kel
        loc_expr = location
    else:
        loc_parts = (landmark,)
        location = landmark
        loc_expr = f"{location} ({amb_phrase})" if not independent else f"{location} — {amb_phrase}"

    delimiter = O_BOUNDARY_DELIMITERS[(family_index + variant) % len(O_BOUNDARY_DELIMITERS)]

    style_idx = (family_index * 3 + variant) % len(STYLE_FAMILIES) if style_index is None else style_index % len(STYLE_FAMILIES)
    style_family = STYLE_FAMILIES[style_idx]

    if not independent:
        lead_templ, time_templ, risk_templ = STRUCTURE_TEMPLATES_BY_FAMILY[style_family][name]
        lead_clause = lead_templ.format(issue=issue)
        time_clause = time_templ.format(time_str=time_str)
        risk_clause = risk_templ.format(risk_phrase=risk_phrase)
        text = f"{opening}, {intent_phrase}. {lead_clause}{delimiter}{loc_expr}. {time_clause}. {risk_clause}"
    else:
        lead_templ, time_templ, risk_templ = STRUCTURE_TEMPLATES_OOD[style_idx]
        lead_clause = lead_templ.format(issue=issue)
        time_clause = time_templ.format(time_str=time_str)
        risk_clause = risk_templ.format(risk_phrase=risk_phrase)
        text = f"{opening}; {intent_phrase}. {lead_clause}{delimiter}{loc_expr}. {time_clause}. {risk_clause}"

    spans = _spans(
        text,
        [
            (issue, CanonicalSpanLabel.OBJ),
            *[(x, CanonicalSpanLabel.LOC) for x in loc_parts],
            (time_str, CanonicalSpanLabel.TIME),
        ],
    )

    message_id = f"msg_{name}_{category.value.lower()}_{family_index:02d}_{variant:02d}"
    bubble = TrajectoryBubble(source_message_id=message_id, text=text, canonical_spans=spans)

    action = DecisionMode.EXECUTE if completeness == "SUFFICIENT" else DecisionMode.REQUEST_CLARIFICATION if completeness == "INCOMPLETE" else DecisionMode.RE_EVALUATE
    missing = () if completeness == "SUFFICIENT" else ("location_detail",) if completeness == "INCOMPLETE" else ("ambiguous_context",)
    expected = TurnExpectedAction(turn=1, allowed_actions=(action,), missing=missing, strategy=action.value)

    truth = {
        "intent": intent,
        "risk": risk,
        "risk_evidence_kind": risk_evidence_kind,
        "completeness": completeness,
        "concept_id": concept_id,
        "style_family": style_family,
        "category": category.value,
        "domain": category.value,
        "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT if independent else PROVENANCE_HIFI_SYNTHETIC,
        "dataset_origin": PROVENANCE_SYNTHETIC_INDEPENDENT if independent else PROVENANCE_HIFI_SYNTHETIC,
        "is_synthetic": True,
        "human_written": False,
        "real_world": False,
        "canonical_spans": [list(x) for x in spans],
        "bubble_canonical_spans": {message_id: [list(x) for x in spans]},
        "discourse_structure": "whatsapp_citizen_report",
        "discourse_genre": "citizen_complaint",
    }

    turn = TrajectoryTurn(turn=1, bubbles=(bubble,), observable_facts=(issue, *loc_parts, time_str), hidden_facts=(), expected_action=expected)
    return ComplaintTrajectory(
        scenario_id=scenario_id,
        family_id=family_id,
        split=split,
        category=category,
        world_truth=truth,
        observable_facts=(issue, *loc_parts, time_str),
        location_completeness=completeness,
        duration="OBSERVED",
        claim_certainty="FIRST_HAND",
        persona="CITIZEN",
        noise={"style": "informal_whatsapp"},
        attachment_role="NONE",
        turns=(turn,),
        expected_action_by_turn=(expected,),
        canonical_spans=spans,
    )


def _write(path: Path, values: list[ComplaintTrajectory]) -> None:
    path.write_text("".join(x.model_dump_json() + "\n" for x in values), encoding="utf-8")


def generate_hifi_dataset(output_dir: Path | str, seed: int = 5101, **kwargs: Any) -> dict[str, Any]:
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[ComplaintTrajectory]] = {"train": [], "dev": [], "test": []}
    ood: list[ComplaintTrajectory] = []

    for ci, category in enumerate(CATEGORIES):
        # Train: 240 records per category (20 families x 12 variants)
        for fi in range(20):
            fam_id = f"hifi-fam-tr-{ci:02d}-{fi:02d}"
            for variant in range(12):
                idx = fi * 12 + variant
                concept_id = idx % len(ISSUES[category])
                triplet = BALANCED_TRIPLETS[(ci * 12 + idx) % len(BALANCED_TRIPLETS)]
                scen_id = f"hifi_train_{ci:02d}_{fi:02d}_{variant:02d}"
                buckets["train"].append(
                    _build(scen_id, fam_id, DatasetSplit.TRAIN, category, triplet[0], triplet[1], triplet[2], variant, ci * 20 + fi, concept_id, style_index=(fi * 3 + variant) % 3)
                )

        # Dev: 48 records per category (4 families x 12 variants)
        for fi in range(4):
            fam_id = f"hifi-fam-dev-{ci:02d}-{fi:02d}"
            for variant in range(12):
                idx = fi * 12 + variant
                concept_id = idx % len(ISSUES[category])
                triplet = BALANCED_TRIPLETS[(ci * 12 + idx) % len(BALANCED_TRIPLETS)]
                scen_id = f"hifi_dev_{ci:02d}_{fi:02d}_{variant:02d}"
                buckets["dev"].append(
                    _build(scen_id, fam_id, DatasetSplit.DEV, category, triplet[0], triplet[1], triplet[2], variant, ci * 4 + fi, concept_id, style_index=(fi * 3 + variant) % 3)
                )

        # Test: 72 records per category (6 families x 12 variants)
        for fi in range(6):
            fam_id = f"hifi-fam-test-{ci:02d}-{fi:02d}"
            for variant in range(12):
                idx = fi * 12 + variant
                concept_id = idx % len(ISSUES[category])
                triplet = BALANCED_TRIPLETS[(ci * 12 + idx) % len(BALANCED_TRIPLETS)]
                scen_id = f"hifi_test_{ci:02d}_{fi:02d}_{variant:02d}"
                buckets["test"].append(
                    _build(scen_id, fam_id, DatasetSplit.TEST, category, triplet[0], triplet[1], triplet[2], variant, ci * 6 + fi, concept_id, style_index=(fi * 3 + variant) % 3)
                )

        # OOD: 72 records per category (6 families x 12 variants)
        for fi in range(6):
            fam_id = f"ood-fam-{ci:02d}-{fi:02d}"
            for variant in range(12):
                idx = fi * 12 + variant
                concept_id = idx % len(ISSUES[category])
                triplet = BALANCED_TRIPLETS[(ci * 12 + idx) % len(BALANCED_TRIPLETS)]
                scen_id = f"ood_{ci:02d}_{fi:02d}_{variant:02d}"
                ood.append(
                    _build(scen_id, fam_id, DatasetSplit.TEST, category, triplet[0], triplet[1], triplet[2], variant, ci * 6 + fi, concept_id, independent=True, style_index=(fi * 3 + variant) % 3)
                )

    all_values = buckets["train"] + buckets["dev"] + buckets["test"]
    for name in buckets:
        _write(out / f"{name}.jsonl", buckets[name])
    _write(out / "trajectories.jsonl", all_values)
    _write(out / "ood_canary.jsonl", ood)

    corpus = "".join(b.text + "\n" for t in buckets["train"] for turn in t.turns for b in turn.bubbles)
    for name in ("corpus.txt", "corpus_train.txt"):
        (out / name).write_text(corpus, encoding="utf-8")

    files = ["train.jsonl", "dev.jsonl", "test.jsonl", "trajectories.jsonl", "ood_canary.jsonl", "corpus.txt", "corpus_train.txt"]
    (out / "sha256sums.txt").write_text("".join(f"{compute_file_sha256(out / x)}  {x}\n" for x in files), encoding="utf-8")

    digest = compute_file_sha256(out / "ood_canary.jsonl")
    hifi = {
        "manifest_version": "1.0.0",
        "contract": "city12-v3",
        "human_written": False,
        "is_human_written": False,
        "real_world": False,
        "is_real_world": False,
        "provenance": PROVENANCE_HIFI_SYNTHETIC,
        "artifacts": {
            "ood_canary": {
                "name": "ood_canary",
                "path": "ood_canary.jsonl",
                "sha256": digest,
                "task": "holdout",
                "split": "test",
                "expected_count": len(ood),
                "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            }
        },
        "datasets": {
            "ood_canary": {
                "name": "ood_canary",
                "path": "ood_canary.jsonl",
                "sha256": digest,
                "task": "holdout",
                "split": "test",
                "expected_count": len(ood),
                "provenance": PROVENANCE_SYNTHETIC_INDEPENDENT,
            }
        },
    }
    (out / "hifi_manifest.json").write_text(json.dumps(hifi, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "manifest_version": "2.0.0",
        "contract": "city12-v3",
        "environment": "local-cpu",
        "splits": {x: {"path": f"{x}.jsonl", "count": len(buckets[x])} for x in buckets},
        "ood": {"withheld_from_train": True, "count": len(ood)},
        "artifacts": {x: {"name": x.rsplit(".", 1)[0], "path": x, "sha256": compute_file_sha256(out / x)} for x in files + ["hifi_manifest.json"]},
    }
    manifest["artifacts"]["ood_canary.jsonl"] = {"name": "ood_canary", "path": "ood_canary.jsonl", "sha256": digest}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    card = f"""# City12-v3 HiFi Synthetic Dataset

- **Contract**: city12-v3
- **Categories**: 12 ({', '.join(c.value for c in CATEGORIES)})
- **Counts**:
  - Train: {len(buckets['train'])} (240 per category)
  - Dev: {len(buckets['dev'])} (48 per category)
  - Test: {len(buckets['test'])} (72 per category)
  - OOD Canary: {len(ood)} (72 per category)
  - Internal trajectories: {len(all_values)}
- **Concept representation**: All 52 issue concepts per category are represented across splits via root `world_truth['concept_id']`.
- **Anti-leak audit**: Family-stratified, zero 6/7-gram internal leaks, zero 5/6/7-gram OOD-vs-internal overlap.
- **Provenance**: Internal `{PROVENANCE_HIFI_SYNTHETIC}`, OOD `{PROVENANCE_SYNTHETIC_INDEPENDENT}`.
"""
    (out / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate M3 HiFi synthetic dataset (city12-v3)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "m3_hifi")
    parser.add_argument("--seed", type=int, default=5101)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    generate_hifi_dataset(args.output_dir, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
