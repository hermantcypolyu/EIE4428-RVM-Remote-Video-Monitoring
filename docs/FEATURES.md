# RVM Extended Features (扩展功能)

本文档介绍RVM系统新增的高级监控功能。

## 功能列表

### 1. 目标跟踪 (Object Tracking)

使用质心追踪算法为检测到的物体分配持久ID，支持跨帧跟踪。

**特性:**
- 基于IoU (Intersection over Union) 的匹配算法
- 保留轨迹历史记录
- 自动处理物体消失和重新出现
- 实时性能优化

**使用示例:**
```python
from rvm.analysis.track import CentroidTracker

tracker = CentroidTracker(max_disappeared=10, iou_threshold=0.3)

# 每帧更新
detections = [(top, left, bottom, right), ...]  # 检测框列表
tracked = tracker.update(detections)

# 访问跟踪结果
for track_id, track in tracked.tracks.items():
    print(f"Track {track_id}: {track.centroid}, Age: {track.age}")
```

### 2. 人脸识别 (Face Recognition)

超越人脸检测，识别已知人员身份。

**特性:**
- 基于深度学习嵌入向量
- 支持DeepFace和dlib后端
- 可注册已知人脸数据库
- 优雅降级（模型不可用时）

**使用示例:**
```python
from rvm.analysis.recognition import FaceRecognizer

recognizer = FaceRecognizer(threshold=0.6)

# 注册已知人脸
recognizer.add_known_face("张三", embedding_vector)

# 识别人脸
face_boxes = [(x, y, w, h), ...]
result = recognizer.recognize(frame, face_boxes)

for name, confidence, bbox in result.identities:
    print(f"识别到 {name}，置信度 {confidence:.2f}")
```

### 3. 情绪识别 (Emotion Recognition)

从面部表情分类情绪状态。

**特性:**
- 支持7种基本情绪：愤怒、厌恶、恐惧、快乐、悲伤、惊讶、中性
- 使用DeepFace或FER模型
- 实时处理能力
- 为每个检测到的人脸提供情绪标签

**使用示例:**
```python
from rvm.analysis.recognition import EmotionRecognizer

recognizer = EmotionRecognizer()

face_boxes = [(x, y, w, h), ...]
result = recognizer.detect(frame, face_boxes)

for emotion, confidence, bbox in result.emotions:
    print(f"情绪: {emotion}，置信度 {confidence:.2f}")
```

### 4. 手势识别 (Gesture Recognition)

检测和分类手部姿势，用于非接触控制和交互监控。

**特性:**
- 使用MediaPipe Hands进行手部关键点检测
- 识别常见手势：张开手掌、握拳、指向、竖大拇指等
- 区分左右手
- 提供手部边界框和关键点

**使用示例:**
```python
from rvm.analysis.gesture import GestureRecognizer, GestureType

recognizer = GestureRecognizer(max_hands=2)

detection = recognizer.detect(frame)

for gesture in detection.gestures:
    print(f"{gesture.hand_label}手: {gesture.gesture.value}, "
          f"置信度 {gesture.confidence:.2f}")
```

**支持的手势:**
- `OPEN_PALM` - 张开手掌 (停止/警告)
- `CLOSED_FIST` - 握拳
- `POINTING` - 指向
- `THUMBS_UP` - 竖大拇指
- `THUMBS_DOWN` - 倒大拇指
- `PEACE` - 和平手势(V字)
- `OK_SIGN` - OK手势

### 5. 音频监控 (Audio Monitoring)

音频捕获和事件检测，完善A/V监控系统。

#### 5.1 音频捕获

**特性:**
- 跨平台麦克风访问
- 可配置采样率和缓冲区大小
- 独立线程运行，不阻塞视频管道

**使用示例:**
```python
from rvm.audio import AudioCapture, AudioConfig

config = AudioConfig(
    sample_rate=16000,
    channels=1,
    chunk_size=1024
)

capture = AudioCapture(config)
capture.start()

# 读取音频
while True:
    audio_chunk = capture.read(timeout=0.1)
    if audio_chunk is not None:
        # 处理音频数据
        pass

capture.stop()
```

#### 5.2 音频事件分析

**特性:**
- 检测响亮声音（警报、玻璃破碎、呼喊）
- 枪声检测
- 异常静音检测
- 语音活动检测（VAD）
- 基于信号处理，无需深度学习模型

**使用示例:**
```python
from rvm.audio import AudioAnalyzer, EventType

analyzer = AudioAnalyzer(
    sample_rate=16000,
    energy_threshold=0.02,
    silence_threshold=0.005
)

audio_chunk = capture.read()
events = analyzer.analyze(audio_chunk)

for event in events:
    print(f"事件: {event.event_type.value}, "
          f"时间: {event.timestamp:.2f}s, "
          f"能量: {event.energy:.3f}")
```

**检测的事件类型:**
- `LOUD_SOUND` - 响亮声音
- `SPEECH` - 语音
- `SILENCE` - 静音
- `GUNSHOT` - 枪声
- `GLASS_BREAK` - 玻璃破碎
- `ALARM` - 警报

## 系统集成

所有新功能都设计为：
1. **非阻塞** - 在独立线程或异步运行，不影响视频编码
2. **优雅降级** - 依赖模型不可用时返回空结果而不是失败
3. **可选启用** - 所有扩展功能都是可选的，核心系统无需这些依赖即可运行
4. **JSON友好** - 所有结果都提供 `.snapshot()` 方法用于Web API

## 性能考虑

| 功能 | CPU占用 | 延迟 | 推荐配置 |
|------|---------|------|----------|
| 目标跟踪 | 低 | <5ms | 始终启用 |
| 人脸识别 | 中-高 | 50-200ms | 按需启用 |
| 情绪识别 | 中-高 | 50-200ms | 按需启用 |
| 手势识别 | 中 | 10-50ms | 实时交互场景 |
| 音频分析 | 低 | <10ms | 始终启用 |

## 依赖安装

核心功能只需要：
```bash
pip install numpy aiohttp opencv-python-headless
```

启用扩展功能：
```bash
# 人脸识别和情绪识别（MediaPipe）
pip install mediapipe

# 手势识别（MediaPipe）
pip install mediapipe

# 音频
pip install pyaudio
```

## 未来扩展

计划添加的功能：
- 行为分析（异常行为检测）
- 多目标重识别（Re-ID）
- 场景理解和活动识别
- 音频流式传输和编码
- 双向音频通信

## 测试

运行扩展功能的测试：
```bash
pytest tests/test_track.py      # 目标跟踪
pytest tests/test_gesture.py    # 手势识别
pytest tests/test_audio.py      # 音频
```

## 许可和引用

本项目使用的第三方库：
- MediaPipe (Apache 2.0)
- DeepFace (MIT)
- PyAudio (MIT)

详细文档请参考各库的官方文档。
