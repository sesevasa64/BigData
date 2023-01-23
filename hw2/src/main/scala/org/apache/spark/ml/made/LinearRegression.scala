package org.apache.spark.ml.made

import breeze.numerics.{abs,sqrt}
import breeze.linalg.{DenseVector, InjectNumericOps, norm}
import breeze.stats.distributions.Uniform
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param.{DoubleParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol, HasMaxIter, HasTol}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}

trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol
  with HasPredictionCol with HasMaxIter with HasTol {
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setTol(value: Double): this.type = set(tol, value)

  private val learningRate = new DoubleParam(
    this, "learningRate", "")

  def getLearningRate: Double = $(learningRate)
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  setDefault(learningRate -> 0.0001)
  setDefault(maxIter -> 200)
  setDefault(tol -> 0.001)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, DoubleType)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getLabelCol).copy(name = getPredictionCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder: Encoder[(Vector, Double)] = ExpressionEncoder()

    val vectors = dataset.select(dataset($(featuresCol)), dataset($(labelCol)).as[(Vector, Double)])

    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(featuresCol)))).numAttributes.getOrElse({
      val row = vectors.first()
      val x= row.getAs[Vector]($(featuresCol))
      x.size
    })

    val rows = vectors.rdd

    val uniform_gen = Uniform(-1 / sqrt(dim), 1 / sqrt(dim))
    var weights = DenseVector.rand(dim, rand=uniform_gen)
    var bias = uniform_gen.get()
    var training_done = false

    for (i <- 1 to $(maxIter); if !training_done) {
      val (w_grad, b_grad) = rows.map(row => {
        val x = row.getAs[Vector]($(featuresCol)).asBreeze
        val y_true = row.getAs[Double]($(labelCol))
        val y_pred = weights.dot(x) + bias
        val eps = y_pred - y_true
        (x *:* eps, eps)
      }).reduce((acc, curVal) => (acc._1 + curVal._1, acc._2 + curVal._2))
      val weights_new = weights - getLearningRate *:* w_grad
      val bias_new = bias - getLearningRate * b_grad
      if (norm(weights_new - weights) + abs(bias_new - bias) < $(tol))
        training_done = true
      weights = weights_new
      bias = bias_new
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights), bias))
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made] (override val uid: String, val weights: Vector, val bias: Double)
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[made] def this(weights: Vector, bias: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bWeights = weights.asBreeze
    val transformUdf = {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => {
          bWeights.dot(x.asBreeze) + bias
        })
    }
    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val coefs = (weights, bias)

      sqlContext.createDataFrame(Seq(coefs)).write.parquet(path + "/coefs")
    }
  }
}


object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val coefs = sqlContext.read.parquet(path + "/coefs")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val vector_encoder: Encoder[Vector] = ExpressionEncoder()
      // Used to convert untyped dataframes to datasets with doubles
      implicit val double_encoder: Encoder[Double] = ExpressionEncoder()

      val (weights, bias) =  coefs.select(coefs("_1").as[Vector], coefs("_2").as[Double]).first()

      val model = new LinearRegressionModel(weights, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}
