﻿namespace MLExperiments

module SpamOrHamBayes = 

    // https://github.com/dotnet/machinelearning-samples/tree/master/samples/fsharp/getting-started/BinaryClassification_SpamDetection

    open System
    open System.IO
    open Microsoft.ML
    open Microsoft.ML.Data
    open System.Net
    open System.IO.Compression
    open Microsoft.ML.Transforms.Text

    [<CLIMutable>]
    type SpamInput = 
        {
            [<LoadColumn(0)>]
            Label : string
            [<LoadColumn(1)>]
            Message : string
        }
    
    [<CLIMutable>]
    type SpamPrediction = 
        {
            PredictedLabel : string
        }
    
    let downcastPipeline (x : IEstimator<_>) = 
        match x with 
        | :? IEstimator<ITransformer> as y -> y
        | _ -> failwith "downcastPipeline: expecting a IEstimator<ITransformer>"

    let classifyWithThreshold threshold (p : PredictionEngine<_,_>) x = 
        let prediction = p.Predict({Label = ""; Message = x})
        printfn "The message '%s' is %s" x (if prediction.PredictedLabel = "spam" then "spam" else "not spam")

    let runExample () =

        let trainDataPath = Path.Combine(Data.root, "SMSSpamCollection")

        let mlContext = MLContext(seed = Nullable 1)
    
        let data = mlContext.Data.LoadFromTextFile<SpamInput>(path = trainDataPath, hasHeader = true, separatorChar = '\t')

        // Create the estimator which converts the text label to boolean, featurizes the text, and adds a linear trainer.
        // Data process configuration with pipeline data transformations 
        let dataProcessPipeline =
            EstimatorChain()
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Label"))
                .Append(mlContext.Transforms.Text.FeaturizeText("FeaturesText", TextFeaturizingEstimator.Options
                                  (
                                      WordFeatureExtractor = WordBagEstimator.Options(NgramLength = 2, UseAllLengths = true),
                                      CharFeatureExtractor = WordBagEstimator.Options(NgramLength = 3, UseAllLengths = false)
                                  ), "Message"))
                .Append(mlContext.Transforms.CopyColumns("Features", "FeaturesText"))
                .Append(mlContext.Transforms.NormalizeLpNorm("Features", "Features"))
                .AppendCacheCheckpoint(mlContext)

        let trainer = 
            EstimatorChain()
                .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName = "Label", featureColumnName = "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"))
        
        let trainingPipeLine = dataProcessPipeline.Append(trainer)

        // Evaluate the model using cross-validation.
        // Cross-validation splits our dataset into 'folds', trains a model on some folds and 
        // evaluates it on the remaining fold. We are using 5 folds so we get back 5 sets of scores.
        // Let's compute the average AUC, which should be between 0.5 and 1 (higher is better).
        printfn "=============== Cross-validating to get model's accuracy metrics ==============="
        let crossValidationResults = mlContext.MulticlassClassification.CrossValidate(data = data, estimator = downcastPipeline trainingPipeLine, numberOfFolds = 5);
    
        let model = trainingPipeLine.Fit(data)

        // The dataset we have is skewed, as there are many more non-spam messages than spam messages.
        // While our model is relatively good at detecting the difference, this skewness leads it to always
        // say the message is not spam. We deal with this by lowering the threshold of the predictor. In reality,
        // it is useful to look at the precision-recall curve to identify the best possible threshold.
        let classify = classifyWithThreshold 0.15f
    
        // Create a PredictionFunction from our model 
        let predictor = mlContext.Model.CreatePredictionEngine<SpamInput, SpamPrediction>(model);
    
        printfn "=============== Predictions for below data==============="
        // Test a few examples
        [
            "That's a great idea. It should work."
            "free medicine winner! congratulations"
            "Yes we should meet over the weekend!"
            "you win pills and free entry vouchers"
        ] 
        |> List.iter (classify predictor)

        printfn "=============== End of process, hit any key to finish =============== "
