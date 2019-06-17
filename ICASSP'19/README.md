# Instrument streaming

This repo contains the instrument streaming model presented in the paper:
[Yun-Ning Hung and Yi-Hsuan Yang, "MULTITASK LEARNING FOR FRAME-LEVEL INSTRUMENT RECOGNITION"](https://arxiv.org/pdf/1811.01143.pdf)

### Instrument categories
1: Piano / 2: Acoustic Guitar / 3: Electrical Guitar / 4: Trumpet / 5: Saxophone / 6: Bass / 7: Violin / 8: Cello / 9: Flute

### Demo
Related websites: 
- [Instrument Activity Detection](https://biboamy.github.io/instrument-recognition/demo.html)

## Run the prediction
1. Put MP3/WAV files in the "mp3" folder
2. Run the 'prediction.py' with the name of the song as the first arg
```
python3 prediction.py make_you_feel_my_love.mp3
```
3. Instrument prediction result will be stored in the **output_data** folder 
