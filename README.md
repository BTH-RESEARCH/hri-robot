# hri-robot

Robot hỗ trợ **phụ huynh** chơi và dạy **trẻ nhỏ chưa biết nói**: phụ huynh ra lệnh bằng giọng nói, trẻ chỉ phản ứng bằng nét mặt. Chạy trên **Raspberry Pi 5** và **laptop** với cùng một codebase. Nguyên mẫu nghiên cứu, luôn dùng dưới sự giám sát của người lớn, **chưa thử nghiệm với trẻ em**.

```
PHỤ HUYNH ─ mic ─► VAD/ASR ─► lệnh ──► INTERVENTION LOGIC ──► LLM + trí nhớ phiên ─► [TOPIC][EMO] câu… ─► TTS + mặt robot
                                              ▲                                                            │
TRẺ ─ camera ─► mặt vùng giữa ─► ResNet-18 ─► hậu xử lý ─► frame queue ─► 4 NHÓM ─► micro-loop: mặt robot   │ robot nói
   ▲                                                          │                                            │
   │                                                          ├─► phản ứng với nội dung đang nghe ─► thích / không thích, nhịp đọc
   └────────────────────── trẻ phản ứng ◄─────────────────────┴─► khó chịu kéo dài: dừng, trấn an, báo phụ huynh
```

- **FER:** ResNet-18 (FERPlus, 8 lớp) chạy cục bộ bằng ONNX Runtime; ảnh không rời thiết bị. Lời phụ huynh (dạng chữ) và nhóm trạng thái của bé có gửi lên LLM.
- **STT / TTS:** sherpa-onnx 2.0.3 (Zipformer tiếng Việt, Piper `vi_VN-vais1000`). Không nhận diện hay lưu giọng nói của người dùng.
- **Trí nhớ:** theo phiên gặp mặt, chỉ nằm trong RAM, xóa khi người dùng rời đi.
- **LLM:** API tương thích OpenAI; mặc định Gemini `gemini-3.5-flash-lite`, có provider dự phòng.
- **Khuôn mặt robot:** đôi mắt phát sáng theo thiết kế `faces/index.html`, vẽ bằng pygame (Pi OS Lite không có trình duyệt); LLM chọn biểu cảm qua thẻ `[EMO:nhãn|cường độ]`.
- **Hai vòng lặp:** lấy cảm hứng từ Fluid-Xpress (HRI '26), đổi cho trường hợp người nói và người được quan sát là hai người.

---

## Mục lục
1. [Yêu cầu](#1-yêu-cầu)
2. [Cài đặt trên laptop](#2-cài-đặt-trên-laptop)
3. [Cài đặt trên Raspberry Pi 5](#3-cài-đặt-trên-raspberry-pi-5)
4. [API key cho LLM](#4-api-key-cho-llm)
5. [Model](#5-model)
6. [Kiểm tra từng thành phần](#6-kiểm-tra-từng-thành-phần)
7. [Chạy: `hri test` và `hri run`](#7-chạy-hri-test-và-hri-run)
8. [Tự chạy khi Pi khởi động](#8-tự-chạy-khi-pi-khởi-động)
9. [Cấu hình](#9-cấu-hình)
10. [Cách hệ thống hoạt động](#10-cách-hệ-thống-hoạt-động)
11. [Dữ liệu ghi lại](#11-dữ-liệu-ghi-lại)
12. [Xử lý sự cố](#12-xử-lý-sự-cố)
13. [Phát triển](#13-phát-triển)

---

## 1. Yêu cầu

| | Laptop | Raspberry Pi 5 |
|---|---|---|
| Hệ điều hành | Windows 10/11, Linux, macOS | Raspberry Pi OS Bookworm 64-bit (**Lite** hoặc Desktop) |
| Python | 3.10 – 3.12 | 3.11 (có sẵn trong OS) |
| Camera | Webcam USB / tích hợp | Pi Camera qua cổng CSI |
| Màn hình | Cửa sổ trên desktop | LCD HDMI (Pi OS Lite: vẽ trực tiếp qua `kmsdrm`) |
| Âm thanh | Mic + loa của máy | Mic USB + loa của màn hình HDMI |
| Mạng | Cần cho LLM và lần tải model đầu | như laptop |

Dung lượng model tải về khoảng 250 MB, lưu trong `~/.cache/hri-robot/models`.

---

## 2. Cài đặt trên laptop

### Windows
Đặt repo ở **đường dẫn ngắn** (ví dụ `C:\dev\hri-robot`). Nếu đường dẫn quá dài, DLL của sherpa-onnx sẽ báo lỗi *"The filename or extension is too long"*.

```powershell
git clone https://github.com/BTH-RESEARCH/hri-robot.git C:\dev\hri-robot
cd C:\dev\hri-robot
powershell -ExecutionPolicy Bypass -File scripts\setup_laptop.ps1
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS
```bash
git clone https://github.com/BTH-RESEARCH/hri-robot.git && cd hri-robot
bash scripts/setup_laptop.sh
source .venv/bin/activate
```

Script setup sẽ tạo `.venv`, cài package (`pip install -e ".[dev]"`), tạo `.env` từ `.env.example`, tải model, rồi in `hri info`.

> Cài tay thì phải thêm nguồn cho `piper_phonemize`, vì gói này không có trên PyPI:
> `pip install -e ".[dev]" -f https://k2-fsa.github.io/icefall/piper_phonemize.html`

---

## 3. Cài đặt trên Raspberry Pi 5

### 3.1 Phần cứng
- Pi Camera cắm vào cổng CSI.
- LCD HDMI cắm vào **HDMI0** (cổng gần chân nguồn USB-C). Cắm màn hình **trước khi bật** Pi.
- Mic USB. Loa: dùng luôn loa của màn hình HDMI (không cần loa rời); vì vậy phải cắm màn hình vào **HDMI0**.
- Nên gắn tản nhiệt chủ động (Active Cooler), vì FER, ASR và TTS cùng chạy trên 4 nhân.

### 3.2 Cài đặt
```bash
git clone https://github.com/BTH-RESEARCH/hri-robot.git ~/hri-robot
cd ~/hri-robot
bash scripts/setup_pi.sh
sudo reboot            # để các nhóm quyền video/render/input/audio có hiệu lực
```
Sau khi khởi động lại:
```bash
cd ~/hri-robot && source .venv/bin/activate
nano .env              # điền API key (mục 4)
hri check display      # vẽ thử lên LCD
hri test               # camera + FER trên LCD
hri run                # chạy robot
```

`setup_pi.sh` làm những việc sau:
- Cài qua apt `python3-picamera2`, `python3-libcamera`, **`python3-pygame`**, PortAudio, ALSA (với `--no-install-recommends` để không kéo theo giao diện desktop).
- Thêm user vào các nhóm `video,render,input,audio`.
- Tạo venv với **`--system-site-packages`** để thấy picamera2 và pygame của apt.
- Cài package; gỡ pygame của pip nếu có.
- Tải model.

### 3.3 Vì sao Pi phải dùng pygame của apt (Pi OS Lite)
Pi OS Lite không có X11/Wayland, nên pygame phải vẽ thẳng ra màn hình qua DRM/KMS bằng driver SDL **`kmsdrm`**. Bản pygame cài qua **pip** cho aarch64 đóng gói một SDL **không có `kmsdrm`** (chỉ có x11/wayland/offscreen), nên không thể mở màn hình. Vì vậy:
- `pyproject.toml` không cài pygame qua pip trên aarch64; bản dùng là `python3-pygame` của Debian.
- `ui.sdl_videodriver: auto` tự chọn `kmsdrm` khi không có `DISPLAY`/`WAYLAND_DISPLAY`. Pi OS Desktop và laptop vẫn dùng driver mặc định.
- Chạy qua SSH được: hình hiện trên LCD gắn vào Pi.

### 3.4 LCD HDMI hiển thị sai độ phân giải hoặc màn hình đen
Nhiều LCD HDMI nhỏ gửi thông tin độ phân giải (EDID) sai. Khi đó ép độ phân giải bằng cách thêm vào **cuối dòng duy nhất** của `/boot/firmware/cmdline.txt` (cách một dấu cách), thay bằng kích thước thật của màn hình:
```
video=HDMI-A-1:800x480@60 consoleblank=0
```
Sau đó `sudo reboot` và kiểm tra lại bằng `hri check display`: dòng `OK: display opened by 'kmsdrm' at 800x480` phải ra đúng kích thước.

---

## 4. API key cho LLM

Sửa file `.env` (không bao giờ commit file này):
```ini
GEMINI_API_KEY=...        # https://aistudio.google.com  (provider chính)
GROQ_API_KEY=             # tuỳ chọn, dự phòng
OPENROUTER_API_KEY=       # tuỳ chọn, dự phòng
```
- Provider được thử theo thứ tự trong `llm.providers`. Provider không có key (hoặc key ngắn hơn 20 ký tự) sẽ bị bỏ qua. Nếu provider đang dùng lỗi **trước khi** trả token đầu tiên, hệ thống chuyển sang provider tiếp theo.
- Xem những model mà key của bạn dùng được: `hri check llm --list-models`.
- **Chỉ dùng model "lite"** cho hội thoại. Model `gemini-3.5-flash` (không lite) "suy nghĩ" trước khi trả lời (6–10 s) và bị cụt câu.
- Chỉ bản tóm tắt cảm xúc dạng số được gửi lên LLM, **không bao giờ gửi ảnh**.

---

## 5. Model

Mọi model được tải tự động, có kiểm tra sha256 nếu manifest có ghi:
```bash
hri models list          # [x] = đã có,  * = profile hiện tại cần
hri models download      # tải các model profile cần
hri models download fer --force   # tải lại một model
```

| Tên | Nguồn | Dùng cho |
|---|---|---|
| `fer` | GitHub Release `BTH-RESEARCH/fer_edge` v0.1.0 → `resnet18-ferplus-fp32.ferpack` | FER (ResNet-18 FERPlus, acc 83.4 %) |
| `fer_int8` (tuỳ chọn) | cùng release → `resnet18-ferplus-int8s.ferpack` | FER bản int8 (acc 81.3 %) |
| `yunet` | OpenCV Zoo | phát hiện khuôn mặt |
| `silero_vad`, `asr_vi` | sherpa-onnx releases | VAD + Zipformer tiếng Việt |
| `tts_vi` | sherpa-onnx releases | Piper `vi_VN-vais1000-medium` (1 giọng) |

- **`.ferpack`** là file zip gồm `model.onnx`, `fer.json` (thông số tiền xử lý) và 500 ảnh test kèm logits tham chiếu. Tiền xử lý được đọc từ `fer.json`, không viết cứng trong code: ảnh xám → 48×48 (BOX) → RGB → 224 (BILINEAR) → chuẩn hóa ImageNet, giống hệt lúc huấn luyện.
- Laptop và Pi5 dùng **cùng bản fp32**. Muốn thử int8: `hri models download fer_int8` rồi `hri run --set vision.fer.model=fer_int8`.
- **Dùng model FER mới:** tạo `.ferpack` bằng notebook export của `fer_edge`, upload lên GitHub Release, rồi sửa `url`/`sha256` của `fer` trong `configs/models.json`.
- Đổi thư mục lưu model: biến môi trường `HRI_MODELS_DIR`.

---

## 6. Kiểm tra từng thành phần

Nên chạy lần lượt khi cài trên máy mới:

| Lệnh | Kiểm tra | Kết quả đúng |
|---|---|---|
| `hri info` | thiết bị, profile, model | mọi model `[x]` |
| `hri check display` | mở được màn hình không (quan trọng trên Pi OS Lite) | `OK: display opened by 'kmsdrm' at WxH` |
| `hri check face` | khuôn mặt robot, lần lượt các biểu cảm | không cần model |
| `hri check fer` | ONNX + tiền xử lý trên máy này khớp PyTorch | `agree=100.0%`, `acc≈83.4%` |
| `hri check camera` | camera | in số FPS |
| `hri check vision --seconds 20` | camera → mặt vùng giữa → FER → 4 nhóm → micro-loop (in chữ) | nhãn và nhóm đổi theo mặt |
| `hri check speaker` | loa, trong lúc cửa sổ robot đang mở | nghe tiếng bíp 1.5 s |
| `hri check mic` | liệt kê thiết bị âm thanh, thu 5 s rồi phát lại | nghe được tiếng mình |
| `hri check asr --seconds 15` | nhận dạng tiếng Việt | in câu bạn nói |
| `hri check tts --text "Xin chào"` | giọng nói | in RTF < 1 và phát âm thanh |
| `hri check llm --text "Hôm nay mình mệt quá"` | LLM + thẻ `[EMO]` | 1–3 câu có thẻ |

---

## 7. Chạy: `hri test` và `hri run`

### `hri test` — kiểm tra FER (không dùng mic, LLM hay TTS)
```bash
hri test                       # mở cửa sổ; trên Pi OS Lite là toàn màn hình LCD
hri test --csv logs/fer.csv    # ghi kết quả từng frame ra CSV
hri test --no-calib            # không học nét mặt bình thường
```
Màn hình hiển thị:
- **Bên trái:** camera (đã lật gương), khung xanh là **vùng giữa**; khung dày có số `#` là mặt đang theo dõi, khung xám là mặt bị bỏ qua.
- **Bên phải:** `thô > hiệu chuẩn > ổn định`, 8 thanh xác suất (vạch trắng là đầu ra thô), 4 nhóm trong 5 s, phản ứng và khuôn mặt robot.
- **Phím tắt:** `c` đo lại baseline · `s` chụp màn hình (`~/.local/share/hri-robot/snapshots/`) · `q`/`Esc` thoát.

Trong **3 giây đầu** thấy mặt, hãy để nét mặt bình thường: hệ thống đo baseline cá nhân từ các frame này. Trước khi đo xong hiện `not calibrated`; bấm `c` để đo lại.

### `hri run` — chạy robot đầy đủ
```bash
hri run
hri run --no-vision            # không dùng camera
hri run --set ui.fullscreen=false
```
Trình tự:
1. Đặt camera **chỉ hướng vào bé**, bé ở giữa khung hình. Robot chào khi thấy mặt, hoặc khi phụ huynh nói.
2. Phụ huynh ra lệnh: "kể cho bé nghe về con mèo", "kể tiếp", "nãy giờ bé thích cái nào".
3. Trong lúc robot nói, hệ thống quan sát bé, ghi chủ đề bé có vẻ thích hoặc không thích và đổi nhịp đọc.
4. Bé có vẻ khó chịu liên tục 4 giây: robot dừng nội dung, trấn an và báo phụ huynh (không qua LLM).

Phím trên cửa sổ robot: `c` đo lại baseline · `n` sang bé khác (xóa trí nhớ, chào lại) · `q`/`Esc` thoát.

### `hri chat` — gõ phím thay cho mic (dùng để thử LLM / prompt)
```bash
hri chat --no-vision --no-tts --fake-emotion sadness   # giả lập nét mặt bé (8 nhãn hoặc 4 nhóm)
hri chat                                               # có camera, gõ phím thay mic
```

### Chọn profile
Profile được tự nhận diện: `pi5` trên Raspberry Pi, `laptop` ở các máy khác. Ép profile bằng `hri -p laptop run`, `hri -p pi5 test`, hoặc biến môi trường `HRI_PROFILE`.

---

## 8. Tự chạy khi Pi khởi động
```bash
mkdir -p ~/.config/systemd/user
cp scripts/hri-robot.service ~/.config/systemd/user/      # sửa đường dẫn nếu repo không nằm ở ~/hri-robot
systemctl --user daemon-reload
systemctl --user enable --now hri-robot
sudo loginctl enable-linger $USER                          # chạy ngay khi boot, không cần đăng nhập
journalctl --user -u hri-robot -f                          # xem log
```
Service đặt sẵn `SDL_VIDEODRIVER=kmsdrm`, dùng được với Pi OS Lite.

---

## 9. Cấu hình

Config được ghép nhiều lớp: `configs/default.yaml` ← `configs/<profile>.yaml` ← `--config file.yaml` ← `--set key=value`.
```bash
hri info --dump                                  # xem config cuối cùng sau khi ghép
hri run --set micro_loop.high_arousal=0.6 --set llm.temperature=0.5
```

Các nhóm tham số chính:

| Nhóm | Nội dung |
|---|---|
| `camera.*` | `backend` (`opencv` / `picamera2` / `file`), `index`, `width`, `height`, `flip_horizontal`, `source` (cho `file`) |
| `vision.face_detector.*` | ngưỡng YuNet, `min_face_size`, `margin` |
| `vision.fer.*` | `model` (khóa trong manifest), `num_threads` |
| `vision.sensitivity.*` | bật/tắt, trọng số độ nhạy theo lớp |
| `vision.baseline.*` | `calib_seconds`, `max_non_neutral`, `neutral_margin`, `dynamic_neutral_suppress`, `class_gains` |
| `vision.hysteresis.*` | `t_min`, `min_dwell_s`, `neutral_bridge_s`, `fade_grace_s`, `transition_dynamics`, ngưỡng theo lớp |
| `vision.smoothing.*` | EMA chỉ để đối chứng, `reset_after` |
| `vision.child_zone.*` | vùng giữa, thời gian giữ và đổi khuôn mặt |
| `micro_loop.*` | chu kỳ, cường độ, `pace` (nhịp đọc theo nhóm) |
| `child.*` | hàng đợi trạng thái, ngưỡng vắng mặt, `distress.*` (khó chịu kéo dài) |
| `audio.*` | thiết bị vào/ra, VAD, ASR, TTS (`speed`, `speaker_id` = giọng đọc của robot) |
| `audio.vad.pre_roll_sec`, `audio.tts.tail_sec` | đệm trước tiếng nói (chống mất từ đầu), im lặng trước khi đóng loa (chống mất chữ cuối) |
| `llm.*` | danh sách provider, `temperature`, `max_tokens`, `sentences_child` / `sentences_parent` |
| `memory.*` | thời gian chờ kết thúc phiên, số lượt giữ nguyên văn, nhịp và độ dài tóm tắt |
| `expression.allowed` | các nhãn `[EMO]` LLM được dùng |
| `safety.*` | từ khóa nguy cơ, câu hỗ trợ khẩn |
| `ui.*`, `test_ui.*` | kích thước, toàn màn hình, màu, `sdl_videodriver` |

**Chạy lại video hoặc thư mục ảnh như một camera** (dùng để demo hoặc kiểm thử):
```bash
hri test --set camera.backend=file --set camera.source=samples/video.mp4
```

---

## 10. Cách hệ thống hoạt động

### Chọn khuôn mặt của bé
Chỉ theo dõi mặt có tâm nằm trong vùng giữa (50% bề ngang), bám cùng một mặt; mặt khác phải ở lại 2 s mới được theo dõi và nhận số mới cùng baseline riêng. **Vùng giữa không chứng minh đó là trẻ.**

### Hậu kỳ FER (kế thừa `fer_resnet_demo` v2, biến thể V3b)
```
camera → YuNet → crop vuông (margin 0.15) → ResNet-18 → softmax
  1. Sensitivity scaling   P'_c = w_c·P_c / Σ w_j·P_j        (neutral ×0.45 … disgust ×3.6)
  2. Baseline cá nhân      P_base đo 3 s mặt tĩnh, kẹp lớp phi-neutral ≤ 0.15 (phần dư dồn về neutral);
                           lớp vượt baseline: p + gain_c·Δ (happiness 1.6 … disgust 3.6), còn lại p·0.5;
                           neutral: p − 0.20 − 2.2·max(Δ phi-neutral) khi có cảm xúc chớm xuất hiện
  3. Hysteresis + 2 luật sinh lý → NHÃN ỔN ĐỊNH
       ngưỡng bắt đầu/duy trì theo lớp (neutral 0.55/0.38 … fear 0.18/0.08), giữ t_min 0.25 s,
       luật 1: thời gian lưu 0.25 s; muốn đổi tiêu cực ↔ tích cực phải qua neutral ≥ 0.25 s,
       luật 2: ma trận rào cản chuyển tiếp (vd sadness → happiness +0.35), ân hạn suy tàn 0.2 s
```
- Code nằm ở `src/hri/vision/postproc.py`, port từ `fer_demo/loss.py`, `calibration.py`, `temporal.py` theo **đúng chuỗi của `runners.py` + `configs/default.yaml`** (sensitivity *trước* baseline, `class_gains` lấy từ config). Ở những chỗ README / `EmotionPostProcessor` của demo nói khác, code `runners.py` được ưu tiên.
- Đã kiểm tra trùng khớp với code demo: 13.500 frame ngẫu nhiên, lệch xác suất ≤ 1e-7, **0 frame lệch nhãn**.
- EMA không còn nằm trong chuỗi, giống demo (`vision.smoothing.enabled: false`). Mỗi khâu đều tắt được (`enabled: false`, `transition_dynamics: false` = V3a) để làm ablation.
- Không kế thừa: pedagogy. Detector: YuNet thay cho YOLO.
- Đo giả lập từng khâu so với EMA: `python scripts/eval_postproc.py`.

### Bốn nhóm (tín hiệu GIÁN TIẾP)

| Nhóm | Nhãn gốc | Mặt robot | Nhịp đọc | LLM |
|---|---|---|---|---|
| `neutral_focus` | neutral | listening | ×1.0 | giữ nhịp |
| `enjoy_curious` | happiness, surprise | happy / curious | ×1.0 | mở rộng điều bé thích |
| `confused_sad` | sadness, fear | empathy / concerned | ×0.85 | chậm lại, trấn an, nói đơn giản hơn |
| `dislike_distracted` | anger, disgust, contempt, **mất mặt** | calm (mất mặt: curious) | ×0.95 | rút ngắn, đổi chủ đề, cho nghỉ |

Ba nhãn anger, disgust, contempt chỉ tính vào `dislike_distracted` khi **model thô** cho tổng ít nhất `child.dislike_min_raw` (0.15); nếu không thì là `neutral_focus`. Hậu xử lý khuếch đại mạnh ba lớp này, nên thiếu điều kiện trên thì 2% disgust thô đã đủ bật nhóm. Điều kiện chỉ ở tầng nhóm: nhãn và xác suất không đổi.

FERPlus không có lớp "bối rối" hay "xao nhãng": hai khái niệm này chỉ được suy ra.

### Vòng quan sát
- LLM gắn thẻ `[TOPIC:con mèo]` cho mỗi câu trả lời. Cửa sổ từ lúc robot bắt đầu nói tới 0.7 s sau khi nói xong cho ra `liked` / `disliked` / `unclear`. Chưa hiệu chuẩn thì luôn là `unclear`.
- Trung bình theo chủ đề thành `LIKES` / `DISLIKES` của phiên, gửi cho LLM ở lượt sau.
- Khó chịu (`confused_sad` hoặc `dislike_distracted` **trên mặt nhìn thấy được, đã hiệu chuẩn**) ≥ 80% của 4 s: dừng lời đang nói, trấn an, báo phụ huynh; nghỉ 60 s trước lần tiếp theo. Mất mặt không kích hoạt.
- Phụ huynh nhắc tự làm hại bản thân (từ khóa): trả lời hỗ trợ tại chỗ, không qua LLM.

### Lượt nói
- Mic **tắt trong suốt lượt nói** của robot (kể cả giữa hai câu) vì chưa có khử tiếng vọng; **chấm xanh** ở góc trên bên phải báo mic đã mở. Phụ huynh chờ robot nói xong.
- Loa mở một lần cho cả lượt và phát thêm im lặng trước khi đóng, vì HDMI và Bluetooth còn giữ âm thanh sau khi driver báo xong.
- Mỗi câu nhận dạng được lấy thêm 0.4 s âm thanh trước lúc VAD phát hiện tiếng nói, không lấn sang câu trước hay lời robot.
- Nói với bé 4 đến 5 câu, nói với ba mẹ 2 đến 3 câu; LLM tự nhận ra đang nói với ai.

### Khuôn mặt robot
Port từ `faces/index.html`: cùng hình học 800×480 (tự co theo màn hình), mí, trái tim, vệt sáng, quầng sáng, chớp mắt, liếc, thở, mở mắt khi khởi động.

| Nguồn | Biểu cảm |
|---|---|
| Giữ nguyên thiết kế | `neutral`, `happy`, `sad`, `surprised`, `love`, `sleepy`, `sleep` |
| Suy ra từ thiết kế | `curious` (mắt `neutral` phóng to nhẹ), `excited` (surprised + happy), `empathy` và `concerned` (sad nhẹ), `calm`, `thinking` (liếc lên), `listening` |
| Không dùng | `angry`, `suspicious` (robot không tỏ vẻ dữ với bé), `confused`. Vẫn có trong `DESIGN` ở `ui/expressions.py` |

Thiết kế không có miệng nên mắt **nhịp theo giọng nói**. Không có ai thì robot ngủ (`sleep`). Xem tất cả: `hri check face`. Chỉnh `ui.face_color`, `ui.glow`, `ui.eye_size`, `ui.scanlines` (mặc định tắt: mắt trơn; `true`: mắt sọc), `ui.idle_motion`.

### Trí nhớ theo phiên
- **Bắt đầu** khi có người trước camera (hoặc có người nói). **Kết thúc** khi không thấy mặt *và* không ai nói trong `memory.idle_sec` (45 s), khi người dùng tạm biệt, hoặc bấm `n`. Không dùng camera thì chỉ tính im lặng (180 s).
- Giữ nguyên văn 6–9 lượt gần nhất (gồm cả lời chào). Đủ 10 lượt, 4 lượt cũ nhất được LLM gộp vào bản tóm tắt khoảng 80 từ, chạy nền sau khi robot nói xong.
- Chỉ nằm trong RAM; hết phiên thì xóa. Log nghiên cứu JSONL vẫn ghi từng lượt (không ghi bản tóm tắt).
- Câu trả lời bị ngắt giữa chừng: câu dở không được đọc và không vào trí nhớ; robot tự gọi lại (tối đa `llm.max_attempts` lần) và **nói tiếp** từ câu đã đọc.
- Giới hạn: hai người đổi chỗ trong chưa đầy 45 s dùng chung một phiên (bấm `n` để tách).

### Tin nhắn gửi LLM mỗi lượt
```
MEMORY: ...                            ← tóm tắt phần cũ của phiên, nếu có
CHILD: {"visible": true, "group": "enjoy_curious", "dominance": 0.8, "trend": "stable", "calibrated": true}
REACTION: {"topic": "con mèo", "verdict": "liked"}   ← phản ứng với lần nói trước
LIKES: con mèo / DISLIKES: con sư tử                 ← trong phiên này
ROBOT_FACE: happy                                    ← micro-loop đang hiện
GUIDANCE: ...                                        ← sau khi robot tự can thiệp, hoặc lệnh quá chung chung và chưa có chủ đề
Phụ huynh: kể tiếp đi
```
LLM trả về các câu có thẻ:
```
[TOPIC:con mèo] [EMO:excited|0.7] Bé ơi, meo meo! [EMO:happy|0.7] Mèo con có cái đuôi dài thật dài.
```
- Câu được tách ngay trong lúc LLM đang trả lời: mỗi câu hoàn chỉnh được gửi sang TTS luôn. Nhãn lạ thành `neutral`, cường độ kẹp trong 0–1.
- System prompt ở `src/hri/llm/prompts.py`: hai vai (nói với bé / nói với phụ huynh), an toàn cho trẻ nhỏ, cách dùng tín hiệu gián tiếp, trí nhớ, định dạng và ví dụ.
- Thêm biểu cảm robot mới: thêm preset vào `src/hri/ui/expressions.py`, thêm nhãn vào `expression.allowed`, rồi xem bằng `hri check face`.

---

## 11. Dữ liệu ghi lại

| File | Nội dung |
|---|---|
| `~/.local/share/hri-robot/logs/session_*.jsonl` | `turn` (lời phụ huynh, trạng thái bé, câu trả lời, độ trễ), `reaction` (phản ứng theo chủ đề), `soothe` (can thiệp) |
| `hri test --csv file.csv` | mỗi frame: nhãn thô, EMA, hiệu chuẩn, ổn định, nhóm, xác suất 8 lớp, thời gian xử lý |
| `~/.local/share/hri-robot/snapshots/` | ảnh chụp bằng phím `s` trong `hri test` |

---

## 12. Xử lý sự cố

| Triệu chứng | Nguyên nhân / cách sửa |
|---|---|
| Pi OS Lite: không mở được cửa sổ, lỗi `kmsdrm not available` | Đang dùng pygame của pip. Chạy `sudo apt install python3-pygame && .venv/bin/pip uninstall -y pygame`, kiểm tra `include-system-site-packages = true` trong `.venv/pyvenv.cfg`, rồi chạy `hri check display` |
| `hri check display` báo lỗi quyền | `sudo usermod -aG video,render,input $USER` rồi đăng nhập lại |
| LCD đen hoặc sai độ phân giải | Thêm `video=HDMI-A-1:WxH@60` vào `cmdline.txt` (mục 3.4) |
| Vẫn không vẽ được lên LCD | Dự phòng: `sudo apt install cage` rồi `cage -- .venv/bin/hri run` |
| `PortAudioError: Device unavailable [-9985]` | Loa HDMI chỉ cho một chương trình mở. Bản cũ để pygame chiếm loa (đã sửa: không khởi tạo `pygame.mixer`). Nếu còn gặp: `sudo fuser -v /dev/snd/*` để xem ai đang giữ, rồi `hri check speaker` |
| `Microphone unavailable` / robot không nghe | `arecord -l` không thấy thiết bị nào: màn hình HDMI không có mic, cần cắm mic USB |
| Vẫn mất chữ cuối câu cuối cùng (Bluetooth, HDMI) | Tăng im lặng trước khi đóng loa: `hri run --set audio.tts.tail_sec=0.9` |
| Vẫn mất từ đầu | Đợi chấm xanh trên mặt robot rồi mới nói. Ghi lại để kiểm tra: `hri check asr --seconds 30 --save asr_debug`, nghe file WAV; tăng `audio.vad.pre_roll_sec` nếu từ đầu không có trong file |
| Muốn phát ra loa khác loa mặc định | Xem danh sách bằng `hri check mic`, rồi đặt `audio.output_device` (một phần tên hoặc số thứ tự) |
| Windows: `DLL load failed ... filename or extension is too long` | Chuyển repo sang đường dẫn ngắn, ví dụ `C:\dev\hri-robot` |
| Không cài được `piper_phonemize` | Cài với `-f https://k2-fsa.github.io/icefall/piper_phonemize.html` |
| TTS không ra tiếng, log báo `phoneme_codepoints is empty` | sherpa-onnx 2.x bỏ espeak-ng; cần cài `piper_phonemize` (code đã dùng sẵn) |
| Pi: picamera2 lỗi `numpy.dtype size changed` | numpy 2 trong venv đè numpy của hệ thống; `pip install "numpy<2"` (pyproject đã ghim sẵn trên aarch64) |
| LLM báo `404` | Tên model đã ngừng phục vụ; xem `hri check llm --list-models` rồi đổi `llm.providers[].model` |
| Groq báo `403 Access denied` | Groq chặn theo vùng / mạng, không phải lỗi code; dùng Gemini hoặc OpenRouter |
| Chạy `hri chat` từ Git Bash / script, robot không bao giờ thấy mặt | Lỗi Windows với stdin dạng pipe (đã xử lý: chế độ chat chờ process vision sẵn sàng rồi mới đọc bàn phím) |
| Robot phản ứng sai cảm xúc ngay từ đầu | Baseline học nhầm; bấm `c` để học lại lúc bé bình thường |

---

## 13. Phát triển

```bash
pytest                     # 90 test, không cần phần cứng
ruff check src tests
```

```
configs/            default.yaml, laptop.yaml, pi5.yaml, models.json
scripts/            setup_pi.sh, setup_laptop.ps1/.sh, hri-robot.service
src/hri/
  cli.py            lệnh `hri` (info, models, check, test, chat, run)
  config.py         config nhiều lớp + --set
  models/           tải / kiểm tra sha256 / giải nén .ferpack và tarball
  camera/           picamera2 | OpenCV | file (video / thư mục ảnh)
  vision/           YuNet, face_tracker.py (mặt vùng giữa), FER ONNX, postproc.py, pipeline.py, process vision
  affect/           child_state.py (frame queue, 4 nhóm, thích/không thích), risk.py
  audio/            VAD + ASR, TTS (Piper + piper_phonemize)
  llm/              client có fallback, prompts, parser [EMO]
  dialogue/         app.py (điều phối, vòng đời phiên), memory.py, micro_loop.py
  ui/               expressions.py, face_renderer.py, test_view.py, display.py (kmsdrm)
tests/
```

Tham khảo: Kang et al., *Fluid-Xpress: Emotion-Aware Dual-Loop Framework for Empathic Facial Reaction in HRI*, HRI Companion '26.
