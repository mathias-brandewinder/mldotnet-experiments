namespace MLExperiments

module App = 

    open System
    open System.IO

    open Microsoft.ML
    open Microsoft.ML.Data
    open Microsoft.ML.Trainers
    open Microsoft.ML.Transforms

    open MLExperiments.WineRegression

    [<EntryPoint>]
    let main argv =

        WineForest.runExample ()

        WineRegression.runExample ()

        //SpamOrHam.runExample ()

        SpamOrHamBayes.runExample ()

        0 // return an integer exit code
