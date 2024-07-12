## A Q&Asysrtem based on RAG (基于RAG的问答系统)

Author：Await Li, KarlJx15, Electron-Neutrino

基于通义千问大语言模型，搭建了RAG(检索增强生成)问答系统。

本项目为大学生课设demo，仍存在不完善之处，欢迎大家交流指正

仅供参考，引用请标注出处

#### 注：两个文件均可独立运行，可根据实际使用需求选择合适的文件运行。

### Batching.py

##### 功能说明

该文件侧重于对问题集进行批处理操作，当有大量的问题需要回答和解决时，运行该文件能直接读取问题集，在回答后将问题和对应答案保存至新的表格中。

##### 使用说明

1.安装相应的依赖库和嵌入式模型，模型名：bge-small-zh-v1.5  下载链接：https://www.modelscope.cn/models/Xorbits/bge-small-zh-v1.5/files。

2.打开Batching.py文件，将第83行的api_key修改为自己的api-key（本项目使用了通义千文的api），可根据实际需要将硬编码的api按环境变量进行配置增加安全性。

3.根据实际场景修改第86行的prompt(本项目中为航空器知识与维修场景相关，故使用机务工程师作为prompt)。

4.可根据参考资料的实际路径修改第117-第124行。

5.修改第174行的问题集的文件路径。

6.修改第175行的输出结果的文件路径(若文件不存在，则会创建一个新的文件)。

7.运行该文件。

在运行该文件时，第一次运行由于需要完成文档的预处理、嵌入向量的生成、外挂知识库的建立等操作，因此耗时相对较长。外挂知识库建立完成后将保存在data文件夹中，该项目提供已建立的外挂知识库，也可以删除重新建立。在外挂知识库建立完成后，再次运行时将会跳过该步骤，降低运行时间。

### UI.py

##### 功能说明

该文件侧重于人机对话的页面，当需要回答的问题较少时，运行该文件更加直观，对话页面更加友好。

##### 使用说明

1.安装相应的依赖库和嵌入式模型：模型名：bge-small-zh-v1.5  下载链接：https://www.modelscope.cn/models/Xorbits/bge-small-zh-v1.5/files。

2.打开Batching.py文件，将第75行的api_key修改为自己的api-key（本项目使用了通义千文的api），可根据实际需要将硬编码的api按环境变量进行配置增加安全性。

3.根据实际场景修改第77行的prompt(本项目中为航空器知识与维修场景相关，故使用机务工程师作为prompt)。

4.可根据参考资料的实际路径修改第106-第113行。

5.运行该文件。

在运行该文件时，第一次运行由于需要完成文档的预处理、嵌入向量的生成、外挂知识库的建立等操作，因此提问第一个问题耗时相对较长。外挂知识库建立完成后将保存在data文件夹中，该项目提供已建立的外挂知识库，也可以删除重新建立。在外挂知识库建立完成后，后续提问时时将会跳过该步骤，降低运行时间。

---

Based on the Tongyi Qianwen LLM, a RAG (Retrieval-Augmented Generation) Q&A system has been built.

This project serves as a demo for a university course project and is still a work in progress. Feedback and suggestions are welcome.

For reference only; please cite the source when quoting.

#### Note: Both files can run independently. Choose the appropriate file based on actual usage needs.

### Batching.py

##### Function Description

This file focuses on batch processing a set of questions. When there are a large number of questions to be answered, running this file will directly read the set of questions and save the questions and corresponding answers to a new table after answering them.

##### Instructions

1.Install the relevant dependency libraries and embedded model. Model name: bge-small-zh-v1.5. Download link: https://www.modelscope.cn/models/Xorbits/bge-small-zh-v1.5/files.

2.Open the Batching.py file and change the API key in line 83 to your own API key (this project uses Tongyi Qianwen's API). For increased security, the hardcoded API can be configured as an environment variable as needed.

3.Modify the prompt in line 86 according to the actual scenario (in this project, it is related to aircraft knowledge and maintenance, so an aircraft engineer is used as the prompt).

4.Modify the paths for the reference materials in lines 117-124 based on actual needs.

5.Change the file path of the question set in line 174.

6.Change the file path of the output results in line 175 (if the file does not exist, a new file will be created).

7.Run the file.

When running this file for the first time, it will take relatively longer as it needs to complete operations such as document preprocessing, embedding vector generation, and external knowledge base establishment. The external knowledge base will be saved in the data folder. This project provides a pre-built external knowledge base, but you can also delete and rebuild it. Once the external knowledge base is established, subsequent runs will skip this step, reducing the running time.

### UI.py

##### Function Description

This file focuses on the human-computer interaction interface. When fewer questions need to be answered, running this file is more intuitive, and the dialog interface is more user-friendly.

##### Instructions

1.Install the relevant dependency libraries and embedded model: model name: bge-small-zh-v1.5. Download link: https://www.modelscope.cn/models/Xorbits/bge-small-zh-v1.5/files.

2.Open the UI.py file and change the API key in line 75 to your own API key (this project uses Tongyi Qianwen's API). For increased security, the hardcoded API can be configured as an environment variable as needed.

3.Modify the prompt in line 77 according to the actual scenario (in this project, it is related to aircraft knowledge and maintenance, so an aircraft engineer is used as the prompt).

4.Modify the paths for the reference materials in lines 106-113 based on actual needs.

5.Run the file.

When running this file for the first time, the initial question will take relatively longer due to document preprocessing, embedding vector generation, and external knowledge base establishment. The external knowledge base will be saved in the data folder. This project provides a pre-built external knowledge base, but you can also delete and rebuild it. Once the external knowledge base is established, subsequent questions will skip this step, reducing the running time.
