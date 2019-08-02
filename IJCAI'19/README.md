# Instrument streaming

This repo contains the instrument streaming model presented in the paper:
Yun-Ning Hung, I-Tung Chiang, Yi-An Chen and Yi-Hsuan Yang, "[Musical Composition Style Transfer via Disentangled Timbre Representation](https://arxiv.org/pdf/1905.13567)", 2019 Proc. Int. Joint Conf. Artificial Intelligence (IJCAI’19)

### Instrument categories
1: Piano / 2: Acoustic Guitar / 3: Electrical Guitar / 4: Trumpet / 5: Saxophone / 6: Bass / 7: Violin / 8: Cello / 9: Flute

## Run the prediction
1. Put MP3/WAV files in the "mp3" folder
2. Run the 'prediction.py' with the name of the song as the first arg and the types of model as the second arg
```
python3 prediction.py ocean.mp3 UnetED
```
3. Instrument prediction result and embedding will be stored in the **output_data** folder 
