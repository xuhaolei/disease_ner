# English named entity recognition of disease: Bert+BiLSTM+CRF<br>
English named entity recognition of disease: Bert+bilstm+CRF<br>
tree.txt contains the project list and the functions of each folder or file. <br>
The pre training model of Bert and the trained model are too large to upload. <br>
To modify parameters, please refer to disease_ner/scripts/config.py<br>
<hr>
It is recommended to start the project with Linux server and GPU training: <br>
GET START!<br>
cd disease_ner<br>
To training: python main.py <br>
To predicting: python predict.py <br>
<hr>
Basic Environment: <br>
python 3.8 <br>
pytorch 1.7.1 + cuda11.0 <br>
pytorch-crf 0.7.2 <br>
<hr>
Result: <br>
Biobert F1 socre is about 85% without engineering skills, and it can reach 87% with flood method and adversarial training. Considering the corpus comes from biological paper, its generalization may be more strong. (https://huggingface.co/alvaroalon2/biobert_diseases_ner) <br>
alvaroalon2/biobert_ diseases_ Ner F1 socre is even up to 97% since the corpus is BC5CDR-diseases and NCBI-diseases. It is recommended to use this one without doubt, but the generalization ability is questionable. (https://huggingface.co/alvaroalon2/biobert_diseases_ner) <br>
Another pretrained model is not ideal. <br>
<br>
<br>
<br>
英文疾病命名实体识别Bert+BiLSTM+CRF<br>
tree.txt中包含了工程目录以及各个文件夹或文件的作用<br>
bert预训练模型以及训练所得模型太大无法上传<br>
相关参数请在disease_ner/scripts/config.py中修改<br>
<hr>
开始项目 建议使用linux服务器，用gpu训练：<br>
cd disease_ner<br>
训练 python main.py <br>
预测 python predict.py <br>
<hr>
基本环境：<br>
python 3.8 <br>
pytorch 1.7.1 + cuda11.0 <br>
pytorch-crf 0.7.2 <br>
<hr>
结果:<br>
biobert不用工程技巧跑出来F1_score在85%左右，用洪泛法以及对抗训练可以达到87%，由于bert的语料是生物类，所以泛化性会比较强(https://huggingface.co/alvaroalon2/biobert_diseases_ner)<br>
alvaroalon2/biobert_diseases_ner由于是语料是BC5CDR-diseases和NCBI-diseases，所以效果特别好，甚至可以达到97%，所以推荐用这个，但泛化能力值得商榷。(https://huggingface.co/alvaroalon2/biobert_diseases_ner)<br>
另外一个效果一般<br>
