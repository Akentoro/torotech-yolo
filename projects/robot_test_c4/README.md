# robot_test_c4 — YOLO 偵測目標

## 物件類別

| ID | Class | 說明 | 特徵 |
|----|-------|------|------|
| 0 | box | 棕色紙盒 | HSV H=8-22, 矩形, 多種尺寸 |
| 1 | red_dot | 紅色角貼 | 圓形, 直徑 ~10mm, 貼在盒子四角 |
| 2 | pcba | PCBA 板 | 綠色 PCB, 矩形, 有元件 |

## 拍攝條件

- 相機: 海康 MV-CA060-10GC (6MP GigE), Eye-in-Hand 朝下
- 高度: Z = 200~600mm (對應不同 FOV)
- 曝光: 固定 25ms
- 解析度: 訓練用 3072x2048 原圖, 推論用 1024x682 (downscaled)
- 背景: 白色泡棉 / 灰色工作台

## 資料收集方式

設備端 CLI:
```bash
vision shot --save dataset/img_001.jpg
```
