namespace MLExperiments

module WineForest = 

    open System
    open System.IO

    open Microsoft.ML
    open Microsoft.ML.Data
    open Microsoft.ML.Trainers
    open Microsoft.ML.Transforms
    open Microsoft.ML.Trainers.FastTree

    [<CLIMutable>]
    type WineDescription = {
        [<LoadColumn(0)>]
        FixedAcidity: float32
        [<LoadColumn(1)>]
        VolatileAcidity: float32
        [<LoadColumn(2)>]
        CitricAcid: float32
        [<LoadColumn(3)>]
        ResidualSugar: float32
        [<LoadColumn(4)>]
        Chlorides: float32
        [<LoadColumn(5)>]
        FreeSulfurDioxide: float32
        [<LoadColumn(6)>]
        TotalSulfurDioxide: float32
        [<LoadColumn(7)>]
        Density: float32
        [<LoadColumn(8)>]
        PH: float32
        [<LoadColumn(9)>]
        Sulphates: float32
        [<LoadColumn(10)>]
        Alcohol: float32
        [<LoadColumn(11)>]
        Quality: float32 
        }

    [<CLIMutable>]
    type WinePrediction = {
        [<ColumnName("Score")>]
        Quality: float32
        }

    let runExample () =

        let downcastPipeline (x : IEstimator<_>) = 
            match x with 
            | :? IEstimator<ITransformer> as y -> y
            | _ -> failwith "downcastPipeline: expecting a IEstimator<ITransformer>"

        let context = MLContext (Nullable 123)

        let trainDataPath = Path.Combine(Data.root, "Wines", "train.csv")
        let testDataPath = Path.Combine(Data.root, "Wines", "test.csv")
        
        let trainingDataView = context.Data.LoadFromTextFile<WineDescription>(trainDataPath, hasHeader = true, separatorChar = ';')
        let testDataView = context.Data.LoadFromTextFile<WineDescription>(testDataPath, hasHeader = true, separatorChar = ';')

        let dataProcessPipeline =
            EstimatorChain()
                .Append(context.Transforms.CopyColumns("Label", "Quality"))
                .Append(context.Transforms.NormalizeMeanVariance("FixedAcidity", "FixedAcidity"))
                .Append(context.Transforms.NormalizeMeanVariance("VolatileAcidity", "VolatileAcidity"))
                .Append(context.Transforms.NormalizeMeanVariance("CitricAcid", "CitricAcid"))
                .Append(context.Transforms.NormalizeMeanVariance("ResidualSugar", "ResidualSugar"))
                .Append(context.Transforms.NormalizeMeanVariance("Chlorides", "Chlorides"))
                .Append(context.Transforms.NormalizeMeanVariance("FreeSulfurDioxide", "FreeSulfurDioxide"))
                .Append(context.Transforms.NormalizeMeanVariance("TotalSulfurDioxide", "TotalSulfurDioxide"))
                .Append(context.Transforms.NormalizeMeanVariance("Density", "Density"))
                .Append(context.Transforms.NormalizeMeanVariance("PH", "PH"))
                .Append(context.Transforms.NormalizeMeanVariance("Sulphates", "Sulphates"))
                .Append(context.Transforms.NormalizeMeanVariance("Alcohol", "Alcohol"))
                .Append(context.Transforms.Concatenate("Features", "FixedAcidity", "VolatileAcidity", "CitricAcid", "ResidualSugar", "Chlorides", "FreeSulfurDioxide", "TotalSulfurDioxide", "Density", "PH", "Sulphates", "Alcohol"))
                .AppendCacheCheckpoint(context)
            |> downcastPipeline

        Common.ConsoleHelper.peekDataViewInConsole<WineDescription> context trainingDataView dataProcessPipeline 5 |> ignore
        Common.ConsoleHelper.peekVectorColumnDataInConsole context "Features" trainingDataView dataProcessPipeline 5 |> ignore

        let trainer = 
            context.Regression.Trainers.FastTree (
                labelColumnName = "Label", 
                featureColumnName = "Features",
                numberOfTrees = 1000
                )

        let modelBuilder = dataProcessPipeline.Append trainer

        let trainedModel = modelBuilder.Fit trainingDataView

        let metrics = 
            let predictions = trainedModel.Transform testDataView
            context.Regression.Evaluate(predictions, "Label", "Score")

        Common.ConsoleHelper.printRegressionMetrics (trainer.ToString()) metrics

        // 8.9;0.84;0.34;1.4;0.05;4;10;0.99554;3.12;0.48;9.1;6
        let testCase = {
            FixedAcidity = 8.9f
            VolatileAcidity = 0.84f
            CitricAcid = 0.34f
            ResidualSugar = 1.4f
            Chlorides = 0.05f
            FreeSulfurDioxide = 4.0f
            TotalSulfurDioxide = 10.0f
            Density = 0.99554f
            PH = 3.12f
            Sulphates = 0.48f
            Alcohol = 9.1f
            Quality = 0.0f 
            }

        let predictor = context.Model.CreatePredictionEngine<WineDescription, WinePrediction>(trainedModel)
        let prediction = predictor.Predict testCase

        printfn "%f" prediction.Quality
