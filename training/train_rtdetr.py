from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')

results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=0.0001,
    optimizer='AdamW',
    warmup_epochs=5,
    augment=True,
    device=0,
    project='runs',
    name='rtdetr_rice_v1',
    amp=True,
    cos_lr=True,
    patience=15,
    workers=4,
    cache=True,
    val=True,
)
