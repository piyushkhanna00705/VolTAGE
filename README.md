# VolTAGE: Volatility forecasting via Text-Audio fusion with Graph convolution networks for Earnings calls

VolTAGE is a mutli-task volatility prediction model that uses cross-modal attention fusion across the text and speech modalities while learning the relational encodings between inter-dependent stocks. It contains the following four modules:

  - Speech and text feature extraction
  - Cross-modal attention fusion for text and audio features
  - Semi-supervised graph convolutional neural network
  - Multi-task volatility prediction using "conditioned" LSTMs

### Requirements

  - For being able to use the code from the files run the command ```pip3 install requirements.txt```

### Speech and text feature extraction

  - The data in the folder ```./audio-feature_extraction/OGdataset``` is incomplete and is only for illustration. The entire data is available [here](https://drive.google.com/file/d/15wtWZvSJicF_Ur2V45lCyCjNJQ7QfXth/view)
  - For the textual feature extraction we use the code [here](https://github.com/abhijeet3922/finbert_embedding)
  - For the audio feature extraction the code is available in the following script which can be run using ```python3 ./audio-feature_extraction/audioExtractionPraat.py```

### Cross-modal attention fusion of text and audio

 - The audio and text features extracted from the above step can be made using the key:value format ```<text_file_name_for_call>:feature_vector```
 - To get the multi-modal attentive embeddings trained on the volatility prediction task run the script using ```python3 ./cross-modal-attn/stockspeech-3-at-attn.py```

### Semi-supervised graph convolutional neural network

 - To extract the [WikiData](https://www.mediawiki.org/wiki/Wikibase/DataModel/JSON) company based relations save all your required entity IDs (or Q values) as in ```./semi-supervised-gcn/relation_extraction/earnings_wiki.csv``` 
 - Then to generate the heterogenous graph and save the adjacency matrix run the script ```python3 ./semi-supervised-gcn/relation_extraction/graph_script.py```
 - Save the graph at ```./semi-supervised-gcn/gcn/data``` and run the python script using the command ```python3 ./semi-supervised-gcn/gcn/StockGCN.py``` to get the saved relational embeddings trained on the attentive multi-modal feature representations. These embeddings will be next used to multi-task volatility prediction using conditioned LSTMs
 
### Multi-task volatility prediction using "conditioned" LSTMs

 - The saved embeddings from the previous step must be saved at ```./multi-task-lstm-conditioning```. These are used as the hidden state for the conditioned LSTM trained on the past volatility values which can be generated from the data [here](https://finance.yahoo.com/)
 - Then to get the final model performance results run the python script using the command ```python3 ./multi-task-lstm-conditioning/stockspeech-final--lstm-cond.py```

### Cite this Paper
Consider citing our work if you use our codebase.
```
@inproceedings{sawhney2020voltage,
  title={VolTAGE: volatility forecasting via text-audio fusion with graph convolution networks for earnings calls},
  author={Sawhney, Ramit and Khanna, Piyush and Aggarwal, Arshiya and Jain, Taru and Mathur, Puneet and Shah, Rajiv},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={8001--8013},
  year={2020}
}
```
