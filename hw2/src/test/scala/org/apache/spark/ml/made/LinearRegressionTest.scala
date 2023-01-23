package org.apache.spark.ml.made

import breeze.linalg._
import breeze.linalg.DenseVector
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest._
import flatspec._
import matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val X: DenseMatrix[Double] = LinearRegressionTest._X
  val W: DenseVector[Double] = LinearRegressionTest._W
  val B: Double = LinearRegressionTest._B
  val Y: DenseVector[Double] = LinearRegressionTest._Y
  val data: DataFrame = LinearRegressionTest._data
  val delta = 0.01

  private def validateModel(model: LinearRegressionModel) = {
    val weights = model.weights
    val bias = model.bias

    weights.size should be(3)

    W(0) should be(weights(0) +- delta)
    W(1) should be(weights(1) +- delta)
    W(2) should be(weights(2) +- delta)

    B should be(bias +- delta)
  }

  "Estimator" should "should produce functional model" in {

    println(Y(0))
    println(data.head)
    println(data.schema)

    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
    val model = estimator.fit(data)
    val weights = model.weights
    val bias = model.bias

    println(s"Weights: $weights")
    println(s"Bias: $bias")

    val res =  model.transform(data)
    val row = res.head
    val y_true = Y(0)
    val y_pred = row.getAs[Double](2)

    println(s"Y_true: $y_true, Y_pred: $y_pred")

    validateModel(model)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateModel(model)
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel])
  }

}

object LinearRegressionTest extends WithSpark {
  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](10000, 3)
  lazy val _W: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)
  lazy val _B: Double = 1.0
  lazy val _Y: DenseVector[Double] = (_X * _W) +:+ _B

  import sqlc.implicits._

  lazy val _data: DataFrame = (for ((x, y) <- _X(*, ::).iterator zip _Y.activeValuesIterator)
    yield (Vectors.fromBreeze(x), y)).toSeq.toDF("features", "label")
}
