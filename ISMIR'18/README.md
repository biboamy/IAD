# Instrument Recognition
Paper
* Yun-Ning Hung and Yi-Hsuan Yang, "[FRAME-LEVEL INSTRUMENT RECOGNITION BY TIMBRE AND PITCH](http://ismir2018.ircam.fr/doc/pdfs/55_Paper.pdf)", International Society for Music Information Retrieval Conference (ISMIR), 2018

The instrument recognition model is trained on MusicNet dataset, which contains 7 kinds of instrument - Piano, Violin, Viola, Cello, Clarinet, Horn and Bassoon.

## Prediction Process
This section is for those who want to load the pre-train model directly for real music testing
1. Put MP3/WAV files in the "mp3" folder
2. Run the prediction python file with the name of the song as the first arg and the model's name as the second arg (model's name can be found in path: data/model/)
```
python prediction.py 1819.wav residual
```
3. Instrument prediction result will be stored in the **output_data** folder 