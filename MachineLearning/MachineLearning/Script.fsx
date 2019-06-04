// source: http://fssnip.net/7Wi/title/Configure-F-Interactive-to-use-MicrosoftMLNet
// script currently not working, WIP

//as of ML.Net 1.0 

#r "netstandard"
#I @"packages\Microsoft.ML.TimeSeries\lib\netstandard2.0"
#I @"packages\Microsoft.ML\lib\netstandard2.0"
#I @"packages\Microsoft.ML.DataView\lib\netstandard2.0"
#I @"packages\Microsoft.ML.StaticPipe\lib\netstandard2.0"
#I @"packages\System.Memory\lib\netstandard2.0"
#I @"packages\System.Runtime.CompilerServices.Unsafe\lib\netstandard2.0"
#I @"packages\Microsoft.ML.Ensemble\lib\netstandard2.0"
#I @"packages\Microsoft.ML.CpuMath\lib\netstandard2.0"
#I @"packages\System.Threading.Tasks.Dataflow\lib\netstandard2.0"

// #I @"packages\Microsoft.ML.FastTree\lib\netstandard2.0"

#r "System.Runtime.CompilerServices.Unsafe.dll"
#r "Microsoft.ML.CpuMath.dll"
#r "Microsoft.ML.Core.dll"
#r "Microsoft.ML.DataView.dll"
#r "Microsoft.ML.Data.dll"
// #r @"Microsoft.ML.FastTree.dll"
#r "Microsoft.ML.StaticPipe.dll"
#r "System.Memory.dll"
#r "Microsoft.ML.Transforms.dll"
#r "Microsoft.ML.Ensemble.dll"
#r "Microsoft.ML.PCA.dll"
#r "Microsoft.ML.TimeSeries.dll"
#r "Microsoft.ML.StandardTrainers.dll"

open System
open System.IO
open System.Threading.Tasks
let path = Environment.GetEnvironmentVariable("path")
let current = __SOURCE_DIRECTORY__
let combine a b = IO.Path.GetFullPath(IO.Path.Combine(a,b))

let path' = 
    path 
    + ";" + combine current @"..\packages\Microsoft.ML.1.0.0\runtimes\win-x64\native"
    + ";" + combine current @"..\packages\Microsoft.ML.Mkl.Redist.1.0.0\runtimes\win-x64\native"
    + ";" + combine current @"..\packages\Microsoft.ML.CpuMath.1.0.0\runtimes\win-x64\native"

Environment.SetEnvironmentVariable("path",path')


// now for actual work

open Microsoft.ML
open Microsoft.ML.Data

let context = MLContext (Nullable 123)

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
    [<ColumnName("Quality")>]
    Quality: float32
    }

let downcastPipeline (x : IEstimator<_>) = 
    match x with 
    | :? IEstimator<ITransformer> as y -> y
    | _ -> failwith "downcastPipeline: expecting a IEstimator<ITransformer>"

let trainDataPath = Path.Combine (__SOURCE_DIRECTORY__, "train.csv")
let trainingDataView = context.Data.LoadFromTextFile<WineDescription>(trainDataPath, hasHeader = true, separatorChar = ';')

let dataProcessPipeline =
    EstimatorChain()
        .Append(context.Transforms.CopyColumns("Label", "Quality"))
        .Append(context.Transforms.Concatenate("Features", "FixedAcidity", "VolatileAcidity", "CitricAcid", "ResidualSugar", "Chlorides", "FreeSulfurDioxide"))//, "TotalSulfurDioxide", "Density", "PH", "Sulfates", "Alcohol"))
        .AppendCacheCheckpoint(context)
    |> downcastPipeline

open Microsoft.ML.Trainers
open Microsoft.ML.Transforms

let trainer = context.Regression.Trainers.Sdca(labelColumnName = "Label", featureColumnName = "Features")

let modelBuilder = dataProcessPipeline.Append trainer

let trainedModel = modelBuilder.Fit trainingDataView
