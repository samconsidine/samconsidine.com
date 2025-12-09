---
title: "The H.264 Encoding and WebRTC Stack"
date: 2024-12-09
author: "Sam Considine"
tags: ["video", "h264", "webrtc", "streaming"]
---

## 1. THE RAW INPUT

A single 1080p frame is just a grid of pixels:

```
1920 × 1080 pixels × 3 bytes (RGB) = 6.2 MB per frame

At 30fps: 186 MB/second = 1.5 Gbps

That's way too much for any network.
```

## 2. COLOR SPACE CONVERSION

First, convert RGB to **YUV (typically NV12 or I420)**:

```
RGB → YUV

Y = Luminance (brightness) - full resolution
U = Chrominance (blue-ish) - half resolution
V = Chrominance (red-ish) - half resolution

Why? Human eyes are more sensitive to brightness than color.
We can subsample color without noticeable quality loss.

1080p NV12: 1920×1080 (Y) + 960×540 (UV) = 3.1 MB per frame
                                            ↑
                                     50% smaller already
```

## 3. DIVIDE INTO MACROBLOCKS

The frame is split into 16×16 pixel blocks:

```
┌────┬────┬────┬────┬────┐
│    │    │    │    │    │
├────┼────┼────┼────┼────┤
│    │    │ MB │    │    │  ← Each box is a 16×16 macroblock
├────┼────┼────┼────┼────┤
│    │    │    │    │    │
└────┴────┴────┴────┴────┘

1080p = 120 × 68 = 8,160 macroblocks per frame
```

## 4. FRAME TYPE DECISION

The encoder decides: **I-frame, P-frame, or B-frame?**

```
I-FRAME (Intra/Keyframe):
┌─────────────────┐
│  Full image     │  Encoded independently
│  No references  │  Large (~50-100KB)
└─────────────────┘

P-FRAME (Predicted):
┌─────────────────┐      ┌─────────────────┐
│  Previous       │ ───► │  Current        │
│  frame          │      │  (differences)  │
└─────────────────┘      └─────────────────┘
                         Small (~5-15KB)

B-FRAME (Bi-directional):
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Previous       │ ───► │  Current        │ ◄─── │  Future         │
└─────────────────┘      │  (differences)  │      └─────────────────┘
                         └─────────────────┘
                         Smallest (~3-8KB)
                         BUT adds latency (must wait for future frame)
```

## 5. MOTION ESTIMATION (P/B frames only)

For each macroblock, find where it "came from" in the reference frame:

```
Reference Frame              Current Frame
┌────────────────┐          ┌────────────────┐
│                │          │                │
│    ┌──┐        │          │          ┌──┐  │
│    │😀│        │   ───►   │          │😀│  │
│    └──┘        │          │          └──┘  │
│                │          │                │
└────────────────┘          └────────────────┘

Motion Vector: (120, 0) - "this block moved 120 pixels right"

Instead of storing the block, store:
- Motion vector: 2 bytes
- Residual (small differences): few bytes
```

## 6. TRANSFORM (DCT)

Each macroblock's residual is transformed using **Discrete Cosine Transform**:

```
Spatial Domain          Frequency Domain
(pixel values)          (DCT coefficients)

┌─────────────┐         ┌─────────────┐
│ 52 55 61 66 │         │ 186  -2   1 │
│ 70 61 64 73 │  DCT    │  12   3  -1 │
│ 63 59 55 90 │  ───►   │   4  -2   0 │
│ 67 61 68 81 │         │   1   0   0 │
└─────────────┘         └─────────────┘

Most energy concentrates in top-left (low frequencies).
Bottom-right values are often near zero (high frequencies).
```

## 7. QUANTIZATION (Lossy Step!)

Divide DCT coefficients by a quantization matrix, round to integers:

```
DCT Coefficients        Quantization Matrix       Result
┌─────────────┐        ┌─────────────┐         ┌─────────────┐
│ 186  -2   1 │        │  16   8   4 │         │  12   0   0 │
│  12   3  -1 │   ÷    │   8   8   4 │    =    │   2   0   0 │
│   4  -2   0 │        │   8   4   8 │         │   0   0   0 │
│   1   0   0 │        │   4   8  16 │         │   0   0   0 │
└─────────────┘        └─────────────┘         └─────────────┘

Higher quantization = more zeros = smaller file = lower quality

This is where bitrate control happens!
- Want smaller file? Increase quantization (more zeros)
- Want better quality? Decrease quantization (keep more detail)
```

## 8. ENTROPY CODING

Convert the quantized coefficients to bits efficiently:

```
CAVLC (simpler, used in Baseline profile):
- Variable length codes
- Common values get short codes

CABAC (better compression, used in Main/High profile):
- Context-adaptive binary arithmetic coding
- ~10-15% better compression
- Slightly slower to decode
```

## 9. NAL UNITS

The encoded data is packaged into **NAL (Network Abstraction Layer) units**:

```
┌──────────────────────────────────────────────────┐
│                    H.264 Stream                   │
├──────┬──────┬──────┬──────┬──────┬──────┬───────┤
│ SPS  │ PPS  │ IDR  │  P   │  P   │  P   │  ...  │
└──────┴──────┴──────┴──────┴──────┴──────┴───────┘

NAL Unit Types:
- SPS (Sequence Parameter Set): Resolution, profile, level
- PPS (Picture Parameter Set): Encoding settings
- IDR (I-frame): Keyframe, resets decoder state
- Non-IDR: P-frames and B-frames
- SEI: Supplemental info (timestamps, etc.)
```

## 10. RTP PACKETIZATION

NAL units are split into RTP packets for network transport:

```
Large NAL unit (30KB IDR frame) with MTU 1200:

┌─────────────────────────────────────────────────┐
│                  NAL Unit (30KB)                 │
└─────────────────────────────────────────────────┘
                        │
                        ▼ FU-A Fragmentation
┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐
│ RTP #1 │ │ RTP #2 │ │ RTP #3 │ ... │ RTP #26│
│ START  │ │  MID   │ │  MID   │     │  END   │
└────────┘ └────────┘ └────────┘     └────────┘
  1200B      1200B      1200B          ~600B


Small NAL units can be aggregated (STAP-A):

┌────────┐ ┌────────┐ ┌────────┐
│ NAL 1  │ │ NAL 2  │ │ NAL 3  │   3 small NALs
│  200B  │ │  300B  │ │  400B  │
└────────┴─┴────────┴─┴────────┘
                │
                ▼ STAP-A Aggregation
┌──────────────────────────────┐
│     Single RTP Packet        │   Combined into one
│         ~900B                │
└──────────────────────────────┘
```

## 11. WEBRTC TRANSPORT

RTP packets go through the WebRTC stack:

```
┌─────────────────────────────────────────────────────────┐
│                     Application                          │
├─────────────────────────────────────────────────────────┤
│  RTP (media)                    RTCP (feedback)         │
│  - Video packets                - PLI (Picture Loss)     │
│  - Audio packets                - FIR (Keyframe Request) │
│  - Sequence numbers             - NACK (Retransmit)      │
│  - Timestamps                   - Sender/Receiver Reports│
├─────────────────────────────────────────────────────────┤
│                      SRTP (encrypted)                    │
├─────────────────────────────────────────────────────────┤
│                      DTLS (key exchange)                 │
├─────────────────────────────────────────────────────────┤
│                      ICE (NAT traversal)                 │
├─────────────────────────────────────────────────────────┤
│                      UDP                                 │
└─────────────────────────────────────────────────────────┘
```

## 12. CLIENT-SIDE DECODE

Reverse the process:

```
RTP Packets
    │
    ▼ Jitter Buffer (reorder, handle loss)
    │
    ▼ Depacketize (reassemble NAL units)
    │
    ▼ Entropy Decode (CAVLC/CABAC)
    │
    ▼ Inverse Quantization
    │
    ▼ Inverse DCT
    │
    ▼ Motion Compensation (add prediction)
    │
    ▼ YUV to RGB
    │
    ▼ Display
```

---

## LATENCY AT EACH STAGE

| Stage | Typical Latency | Notes |
|-------|-----------------|-------|
| Capture | 0-33ms | Depends on camera/source |
| Color convert | 1-5ms | GPU accelerated |
| Motion estimation | 5-20ms | Most CPU/GPU intensive |
| Transform + Quantize | 1-5ms | Fast |
| Entropy coding | 1-5ms | CABAC slightly slower |
| Packetization | <1ms | Trivial |
| Network | 20-150ms | Biggest variable |
| Jitter buffer | 0-50ms | Trade-off: latency vs smoothness |
| Decode | 5-20ms | Hardware accelerated |
| Render | 0-16ms | Display refresh rate |

**Total: ~50-300ms glass-to-glass**

---

# Part 2: How WebRTC Works

WebRTC (Web Real-Time Communication) is a protocol stack for peer-to-peer audio, video, and data transmission directly between browsers/apps without requiring a media server.

## The Core Problem

Two devices on the internet usually can't talk directly because:

1. **NAT (Network Address Translation)** - Your device has a private IP (192.168.x.x), not a public one
2. **Firewalls** - Block unsolicited incoming connections
3. **No discovery mechanism** - How do peers find each other?

WebRTC solves all three.

---

## The Connection Process

```
┌─────────────┐                                    ┌─────────────┐
│   Peer A    │                                    │   Peer B    │
│  (Browser)  │                                    │  (Browser)  │
└──────┬──────┘                                    └──────┬──────┘
       │                                                  │
       │  1. Create Offer (SDP)                          │
       │─────────────────────────────────────────────────►│
       │         (via Signaling Server)                   │
       │                                                  │
       │  2. Create Answer (SDP)                         │
       │◄─────────────────────────────────────────────────│
       │         (via Signaling Server)                   │
       │                                                  │
       │  3. Exchange ICE Candidates                      │
       │◄────────────────────────────────────────────────►│
       │         (via Signaling Server)                   │
       │                                                  │
       │  4. STUN: Discover public IPs                   │
       │◄──────────────► STUN Server ◄───────────────────►│
       │                                                  │
       │  5. Direct P2P Connection (or via TURN)         │
       │◄════════════════════════════════════════════════►│
       │              UDP (media flows)                   │
```

---

## Step 1: Signaling (Out of Band)

WebRTC doesn't define how peers discover each other. You need a **signaling server** (WebSocket, HTTP, carrier pigeon - doesn't matter) to exchange:

- **SDP (Session Description Protocol)** - "Here's what I can send/receive"
- **ICE Candidates** - "Here's how you might reach me"

```
SDP Offer Example (simplified):

v=0
o=- 12345 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=rtpmap:96 H264/90000
a=fmtp:96 profile-level-id=42e01f
a=sendrecv
```

This says: "I want to send/receive H.264 video, payload type 96, at 90kHz clock rate"

---

## Step 2: ICE (Interactive Connectivity Establishment)

ICE finds the best path between peers by gathering **candidates**:

```
Candidate Types (in order of preference):

1. HOST        - Direct local IP (192.168.1.50:54321)
                 Works if peers are on same network

2. SRFLX       - Server Reflexive (via STUN)
                 Your public IP as seen by STUN server
                 Works if NAT allows direct UDP

3. RELAY       - Via TURN server
                 All traffic relayed through server
                 Always works, but adds latency + cost
```

**STUN (Session Traversal Utilities for NAT)**:
```
┌─────────┐                          ┌─────────────┐
│  Peer   │  "What's my public IP?"  │ STUN Server │
│         │─────────────────────────►│             │
│         │                          │             │
│         │  "You're 203.0.113.50"   │             │
│         │◄─────────────────────────│             │
└─────────┘                          └─────────────┘

Now peer knows its public address and can share it as an ICE candidate.
```

**TURN (Traversal Using Relays around NAT)**:
```
When direct connection fails (symmetric NAT, strict firewall):

┌─────────┐         ┌─────────────┐         ┌─────────┐
│ Peer A  │◄───────►│ TURN Server │◄───────►│ Peer B  │
└─────────┘         └─────────────┘         └─────────┘

All media flows through TURN. Works everywhere, but:
- Adds latency (extra hop)
- Costs money (bandwidth)
- Not truly peer-to-peer
```

---

## Step 3: DTLS Handshake

Once ICE finds a path, peers do a **DTLS (Datagram TLS)** handshake over UDP:

```
┌─────────┐                           ┌─────────┐
│ Peer A  │                           │ Peer B  │
└────┬────┘                           └────┬────┘
     │                                     │
     │  ClientHello (with fingerprint)     │
     │────────────────────────────────────►│
     │                                     │
     │  ServerHello + Certificate          │
     │◄────────────────────────────────────│
     │                                     │
     │  Key Exchange                       │
     │◄───────────────────────────────────►│
     │                                     │
     │  ═══ Encrypted Channel Ready ═══    │
```

The certificate fingerprints were exchanged in the SDP, so peers can verify identity.

---

## Step 4: SRTP Media Flow

Media is encrypted with **SRTP (Secure RTP)** using keys derived from DTLS:

```
┌─────────────────────────────────────────────────────┐
│                    SRTP Packet                       │
├──────────────┬──────────────────────┬───────────────┤
│  RTP Header  │   Encrypted Payload  │  Auth Tag     │
│   12 bytes   │    (video/audio)     │   10 bytes    │
└──────────────┴──────────────────────┴───────────────┘

RTP Header contains:
- Sequence number (for reordering)
- Timestamp (for synchronization)
- SSRC (identifies the stream)
- Payload type (codec identifier)
```

---

## Step 5: RTCP Feedback

Alongside media, **RTCP (RTP Control Protocol)** provides feedback:

```
RTCP Packet Types:

SR  (Sender Report)      - "I've sent X packets, Y bytes"
RR  (Receiver Report)    - "I received X%, lost Y packets"
NACK                     - "Please retransmit packet #1234"
PLI (Picture Loss)       - "I lost frames, send a keyframe"
FIR (Full Intra Request) - "Send a keyframe NOW"
REMB                     - "My estimated bandwidth is X bps"
```

This enables:
- Packet loss detection and retransmission
- Bandwidth estimation and adaptation
- Keyframe requests for recovery

---

## Data Channels (SCTP)

WebRTC also supports arbitrary data via **SCTP over DTLS**:

```
┌─────────────────────────────────────────┐
│              Data Channel               │
├─────────────────────────────────────────┤
│  SCTP (Stream Control Transmission)     │
│  - Reliable or unreliable delivery      │
│  - Ordered or unordered                 │
│  - Multiple channels multiplexed        │
├─────────────────────────────────────────┤
│  DTLS (encryption)                      │
├─────────────────────────────────────────┤
│  UDP                                    │
└─────────────────────────────────────────┘

Use cases:
- Game state updates
- Chat messages
- File transfer
- Control commands (like joystick input in teleoperation)
```

---

## The Full Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Application                           │
│         (your video call / streaming app)                │
├───────────────────────┬─────────────────────────────────┤
│      Media Track      │         Data Channel            │
│   (video/audio)       │      (arbitrary data)           │
├───────────────────────┼─────────────────────────────────┤
│        SRTP           │            SCTP                 │
│   (encrypted media)   │    (reliable/unreliable data)   │
├───────────────────────┴─────────────────────────────────┤
│                         DTLS                             │
│                  (key exchange + encryption)             │
├─────────────────────────────────────────────────────────┤
│                          ICE                             │
│              (NAT traversal + path selection)            │
├─────────────────────────────────────────────────────────┤
│                          UDP                             │
│                    (unreliable transport)                │
└─────────────────────────────────────────────────────────┘
```

---

## Why UDP?

TCP would seem safer (guaranteed delivery), but for real-time media:

| TCP | UDP |
|-----|-----|
| Retransmits lost packets | Drops lost packets |
| Head-of-line blocking | No blocking |
| Adds latency on loss | Constant latency |
| Good for files | Good for live media |

A retransmitted video frame that arrives 500ms late is useless - you've already moved on. Better to drop it and show the next frame.

WebRTC builds its own reliability mechanisms (NACK, FEC) on top of UDP when needed, giving fine-grained control over the latency/reliability trade-off.

---

## Typical Latency Breakdown

| Component | Latency |
|-----------|---------|
| ICE negotiation | 200-2000ms (one-time) |
| DTLS handshake | 100-300ms (one-time) |
| Per-packet network | 20-150ms |
| Jitter buffer | 0-50ms |
| **Steady-state total** | **~50-200ms** |
