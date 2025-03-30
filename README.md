# PhotoCloak

PhotoCloak is a Python tool designed to protect images from reverse image search engines and AI-based image recognition systems. By applying sophisticated computer vision and machine learning techniques, PhotoCloak imperceptibly modifies images while preserving visual quality and subject recognition for human viewers.

## Features

- 🛡️ **Anti-Recognition Protection**: Applies state-of-the-art adversarial perturbations to evade AI image recognition
- 🔍 **Subject-Aware Processing**: Uses YOLO and SAM (Segment Anything Model) to intelligently process subjects and backgrounds
- ✨ **Visual Quality**: Preserves visual quality with imperceptible modifications
- 📂 **Batch Processing**: Process entire folders of images efficiently
- 📊 **Adjustable Strength**: Customize protection level based on image content and requirements
- 📝 **EXIF Data Management**: Cleans and modifies metadata to enhance privacy

## How It Works

PhotoCloak combines multiple protection techniques:

1. **Advanced Subject Detection**: Uses YOLO and SAM models to accurately identify people and subjects
2. **Adversarial Perturbations**: Applies frequency domain attacks that confuse AI but remain invisible to humans
3. **Smart Processing**: Applies stronger modifications to subjects or backgrounds as needed
4. **Metadata Protection**: Removes original EXIF data and optionally adds misleading information 

## Installation

```bash
# Clone the repository
git clone https://github.com/Boz0o0/PhotoCloak
cd PhotoCloak

# Install dependencies
pip install -r requirements.txt

# Download required models
# The YOLO model will download automatically on first run
# For SAM model, download from Meta and place in models/ directory:
curl -L -o models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

### Basic Usage

```bash
python PhotoCloak.py input_folder/file output_folder/file --strength 0.1
```

### Choosing Strength Levels

Different types of images may require different strength levels:

- **0.01-0.05**: Minimal protection with excellent visual quality, but not very effective at fooling AI systems and algorithms
- **0.05-0.1**: Balanced protection for personal photos (moderately effective)
- **0.1-0.2**: Strong protection for sensitive images (highly effective)
- **0.3+**: Maximum protection (almost 100% effective against most AI systems) with big visible quality trade-offs

## EXIF Data Management

By default, PhotoCloak replaces original EXIF data with randomized iPhone 11 Pro metadata and locations near Mulhouse, France. This can be:

- **Disabled** entirely by commenting out the `metadata_cleaner` function call
- **Modified** to use different device information by editing the function
- **Extended** to add custom metadata

## Authors & Contributors

- **Enzo LAUGEL** - [Boz0o0](https://github.com/Boz0o0)
- **Gibril BELHAIT** - [gibrilprog](https://github.com/gibrilprog)
- **Maxime ENTZ** - [MaxEntz](https://github.com/MaxEntz)
- **Loic PHILLIPE** - [Loic-ally](https://github.com/Loic-ally)

## Contributing

Contributions are welcome! Here's how you can help improve PhotoCloak:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add some amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

Areas where contributions would be particularly valuable:
- Improving subject detection accuracy
- Adding new adversarial techniques
- Enhancing performance
- Adding support for video files

## License

This project is licensed under the MIT License - see the LICENSE file for details.