# Pipeline Paralel ve Bubble Analizi

> Tenzor paralelliği, matrisi sıralar boyunca çarpır. Pipeline paralelliği, modelini sıralar boyunca, bir sıra başına bölür. Mikrobatçlar boru hattı boyunca akıyor. Başlangıç ve sonundaki boş zaman kabarcaktır; onu en aza indirmek tüm teknedir.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 Track C lessons 42-49
**Time:** ~90 min

## Öğrenme Hedefleri

- Bir dizi modeli N aşamalara ayırıp N sıralar boyunca ileri bir boru hattını simüle edin.
- GPipe programını kullanarak M programı mikrobahtlar ile boru hattı boyunca doldur (sadece ileri, sonra geri) ve kabarcık fraksiyonunu hesaplayın.
- Megatron-LM ve PipeDream'da kullanılan 1F1B programı ile balonları karşılaştırın.
- Etapları savun: Etaplardaki eşit hesaplama, etaplardaki eşit parametreler sayısından daha önemlidir.

## Sorun

Fp16'da 70B parametre modeli sadece 140 GB parametre gerektirir. Hiçbir tüketici GPU'sı tutmuyor. ZeRO-3 sıralar boyunca parametreleri parçaladı, ancak her adım için her sıra tam katmanı toplamak için her katman için log ((N) hops ödüyor. Pipeline paralelliği farklı bir yol alır: modeli N aşamalara ayırıp her sırada bir aşama koyulur. Katman 1'nin ileri kısmı 0 sırada bitirir ve etkinleştirme tensörünü 1 sıraya verir; 1 sırası 2 sırayı yürütür ve 2 sırasına el koyar ve böylece devam eder. Geriye doğru akıyor. Hatıra, her sıra sadece bir aşamayı tutduğu için doğrusal olarak düşer; hesaplama sıralıdır, bu da kabarcık sorunu.

Bubble, boru hattının başlangıcında (en ilk mikrobatçın son aşamaya ulaşmasını bekleyen) ve sonunda (son mikrobatçın tekrar akmasına bekleyen) boşluk süredir. M mikrobaçları ve N aşamalarda aşama başına kabarcık fraksiyonu (N-1) /(M+N-1'dir. M=8, N=4'de bu %27'dir. M=64, N=4'te ise %4,5%. Bir adım başına çok sayıda mikrobatç olduğunda kabarcık küçülür, bu da mikrobatç boyutları anlamına gelir, bu da mikrobatç tasarımını yönlendiren kısıtlama.

## Anlaşım

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### GPipe programı

Bu boru hattını geriye doğru başlamadan önce tüm M mikrobatçlarıyla ileriye doldurun; sonra geriye doğru geriye doğru boşaltın. Her mikrobatçın aktivasyonları geriye kadar tutulmalıdır. Böylece hafıza M ile doğrusal olarak büyür. Ön tarafta M+N-1 döngüleri, geri tarafta başka bir M+N-1 döngüsü vardır. Etap başına yararlı iş 2M döngüdür; aşama başına kabarcık 2 ((N-1) döngüdür. Bubble fraksiyonu (N-1) / ((M+N-1) her ileri ve geri bir zaman birimi alırken. M'yi N'den çok büyük seçmek kabarcıkları gizler.

### 1F1B programı

Interleave: bir mikrobatch'in ileri doğru ilerlemesi son aşamaya ulaştığında, geriye doğru başlayın ve geri akmasına izin verin. Program, her aşamada bir ileri ve bir geri dönüşe değişir. Bubble hala N-1'dir ama aktivasyon hafızası mikrobatç sayımı değil boru hattı derinliğiyle sınırlıdır. Üretim boru hattları 1F1B (Megatron, PipeDream) kullanıyor. Ders öncelikle GPipe uygulamaktadır çünkü daha basit ve 1F1B bir egzersiz olarak.

### Neden Eşitlik Eklentileri Eğlence

Eğer aşama 0 50 ms alır ve aşama 1 100 ms alırsa, her döngü aşama 1'de kapalıdır. Diğer aşamalar aşama 1'nin serbest bırakılmasını bekleyen bir döngü için 50 ms boştur. Aynı parametreler sayımı yanlış bir eksindir: bir transformatörün hesaplaması dikkat artı MLP'den her katman üzerinde egemenlik yapmaktadır ve yerleştirme katmanlarında birçok parametreler vardır ancak az hesaplama yapılır.

### Mikrobatç vs. batç

Bir boru hattı, her biri B boyutundaki M mikrobatçlarını çalışır. Etkin parti boyutu M*B'dir. Bir boru hattı aşamasının sonunda bulunan gradient, M*B örneklerindeki gradienttir. Bubble fraksiyonu M'ye bağlıdır; optimizör M*B'yi görür. M'yi ayarlamak, bir mikropatç hafızası (GPipe için yüksek M ile daha yüksek aktivasyon hafızası) karşılığında bir balon ticareti anlamına gelir.

```figure
cd-pipeline-bubble
```

## Yapın

`code/main.py`Uygulamaları:

- `PipelineStage`Küçük bir ...`nn.Module`Bu bir aşama parametrelerini ve açıklamalarını içerir.`forward(activation)`- Evet .
- `Pipeline(stages, num_microbatches)`: GPipe programını simülasyon aşamaları üzerinde simülasyonlu duvar saati kullanarak simülasyonlu aşamalara düzenler.
- `bubble_fraction(num_stages, num_microbatches)`: kapalı biçim (N-1) /(M+N-1).
- Mikrobatçlık iz ve ölçülen kabarcık fraksiyonunu basan 4 aşamalı bir demo.

Çek şunu:

```bash
python3 code/main.py
```

Çıktı: bir aşama-mikrobatç Gantt grafiği ve kapalı biçim tahminine karşı kabarcık yüzdesi.

## Doğada üretim biçimleri

Üç model, boru hattını gemiye ulaşmak için yeterince paralelleştirir.

**Activation checkpointing pairs with pipeline.**GPipe'de uçuşta olan M mikrobatçları ile, etkinleştirme belleği M çarpı bir mikrobatç olur.

**Stage balance is measured, not assumed.**Üretim ekipleri hedef donanımdaki katman başına gerçek hesaplama (FLOP ve duvar saati) ölçümlerini yapan bir profil geçişini yürütürler, sonra bu ölçümle bölünürler.`--num-layers-per-stage`Bayrak, aşamaların katman başına farklı maliyetleri olduğunda eşit olmayan katman sayımlarına izin vermek için bir listeyi kabul eder.

**Send-recv schedule must avoid deadlock.**Bir boru hattı, her aşama göndermeden önce telde kalıntılar alır. Standart düzeltme, ara vermek: önce eşit sıralama aşamaları gönderir, sonra geri dönüştürür, sonra geri dönüştürür. Ders programları açıkça sıralanır, böylece kalıp görünür.

## Kullan

Üretim biçimleri:

- **Megatron-LM.**Ölçüsünde boru hattı paralelliği için referans. 1F1B kullanır ve tensor + boru hattı + verileri paralel olarak birleştirir.
- **DeepSpeed Pipeline.**ZeRO ile entegre; ZeRO-1 + boru hattı en büyük açık modeller için yaygın bir kombinasyon.
- **PyTorch Pipe.**PyTorch'ın doğası olan boru hattı sarısı, üzerine inşa edilmiş.`torch.distributed.pipeline.sync.Pipe`- Evet .

## Gönder

Ders 80 parçalanmış kontrol noktasında aşama başına parametreler kısımlarını saklar. Ders 81 son-son demo üzerinde DDP + ZeRO + boru hattını oluşturur (ruha göre; demo boru hattını çalıştırma süresi için simüle eder).

## Egzersizler

1. 1F1B uygulamasını yap ve balon fraksiyonunun GPipe'ye eşleştiklerini kontrol et. Ama etkinleştirme hafızası sınırlıdır.
2. Daha derin bir modelde gerçek aşama zamanını profil edin ve duvar saatine göre ölçülen aşamaları yeniden dengeleyin.
3. Pipeline mikrobaçları boyunca gradient birikimini ekleyin ve gradient eşdeğer tam parti önüne gradientine eşittir kontrol edin.
4. Boru hattını etkinleştirme kontrol noktasıyla eşleştirin ve hafıza düşüşünü hesaplama maliyetine karşı ölçün.
5. Boru hattını DDP ile birleştirin (her bir boru hattı sırası bir veri paralel grubu boyunca çoğaltılır) ve 2D programı üzerinden mantık edin.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pipeline | "Model parallel along depth" | One stage per rank, activations flow stage to stage |
| Bubble | "Pipeline idle time" | (N-1) steps at start + end where some stages have no work |
| Microbatch | "Slice of the batch" | One forward/backward unit; bubble shrinks as M grows |
| GPipe | "Fill then drain" | All M forwards before any backward; high activation memory |
| 1F1B | "Interleaved schedule" | One forward one backward per stage; bounded activation memory |

## Daha Fazla Okumak

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- Eğitim 76 - programda kullanılan gönderme/içleme primitifleri
- 19 Fase Ders 78 - ZeRO, boru hattına ortogonal ve sıklıkla birleştirilmiştir.
