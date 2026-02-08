✅ Project Structure
plant_methodology/
│
├── config.py
├── utils.py
├── dataset.py
│
├── models/
│   ├── ie_gan.py
│   ├── moe.py
│   └── dinov2_wrapper.py
│
├── optim/
│   └── hgwos.py
│
├── scripts/
│   ├── train_ie_gan.py
│   ├── generate_synthetic.py
│   ├── extract_dinov2_features.py
│   ├── run_hgwos_feature_selection.py
│   └── train_moe.py
│
└── main_pipeline.py
