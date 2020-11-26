-- NOTE: Requires Apache MADLib to be installed in the database
-- Table to hold the training inputs and output
CREATE TABLE public.housing
(
    crim double precision NOT NULL,
    zn double precision NOT NULL,
    indus double precision NOT NULL,
    chas double precision NOT NULL,
    nox double precision NOT NULL,
    rm double precision NOT NULL,
    age double precision NOT NULL,
    dis double precision NOT NULL,
    rad double precision NOT NULL,
    tax double precision NOT NULL,
    ptratio double precision NOT NULL,
    b double precision NOT NULL,
    lstat double precision NOT NULL,
    medv double precision NOT NULL
)

TABLESPACE pg_default;

ALTER TABLE public.housing
    OWNER to postgres;

-- Create, train and test the model
DROP TABLE IF EXISTS housing_linregr, housing_linregr_summary;
SELECT madlib.linregr_train( 'housing',
                             'housing_linregr',
                             'medv',
                             'ARRAY[1, crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]'
                           );

-- Get the predictions, along with the difference in value and root mean squared
-- error for the entire set. Thanks to Vik Fearing for helping me fix a couple
-- of issues with this query!
SELECT housing.*,
    predict,
    medv - predict AS residual,
    sqrt(avg(power(abs(medv - predict), 2)) OVER ()) AS rmse
FROM housing,
    housing_linregr,
    madlib.linregr_predict(coef,
                           ARRAY[1, crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
                          ) predict;