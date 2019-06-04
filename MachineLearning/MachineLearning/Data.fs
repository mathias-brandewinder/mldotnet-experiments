namespace MLExperiments

module Data =

    open System
    open System.IO

    let root = 
        let appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs().[0])
        Path.Combine(appPath, "../../../../../", "Data")

