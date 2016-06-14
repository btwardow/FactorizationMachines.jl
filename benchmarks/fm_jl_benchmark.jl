using FactorizationMachines
using Base.Test

function main()
    train_filename = ASCIIString(ARGS[1])
    test_filename = ASCIIString(ARGS[2])

    info("Running benchmark tests ...")
    info("Train filename: $train_filename")
    info("Test filename : $test_filename")

    (X_train, y_train) = @time read_libsvm(train_filename)
    info("Train dim: $(size(X_train))")
    (X_test, y_test) = @time read_libsvm(test_filename)
    info("Test dim: $(size(X_test))")

    info("Training model...")
    fm = @time train(X_train, y_train, 
        method = Methods.sgd(num_epochs = Int(10), alpha = 0.1),
        model_params = Models.gauss(num_factors = 4))

    info("Prediction over test dataset...")
    p = @time predict(fm, X_test)

    rmse = sqrt(sum((p - y_test).^2) / length(p))
    info("RMSE on test data: $rmse") 
end

main()
