# Title generations

task: You are provided with a dataset containing article headlines from VC.RU, as well as their corresponding parameters, such as the number of views and the number of likes. Your task is to train the GPT-3.5 model (update: you can take any opensource LLM model) on these headlines + fine-tuning the model so that it can generate headlines based on the provided news data, which will attract a large number of views and likes (likes and views this is a target) The task with an asterisk is to evaluate the number of potential likes and views based on the title.

limitations: Due to a lack of computing power and time, the model for determining likes and views was trained on a very stripped-down dataset and converged after a couple of epochs; accordingly, in the resulting file the model put only those values on which it converged.
