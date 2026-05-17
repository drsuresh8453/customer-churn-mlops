"""
pipeline/sagemaker_pipeline.py — Customer Churn MLOps
Author: Suresh D R | DV Analytics
Run ONCE from SageMaker notebook to set up automated retraining.
"""

import boto3, sagemaker, os
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET',             'customer-churn-project-2024')
REGION         = os.getenv('AWS_REGION',            'ap-south-1')
PIPELINE_NAME  = 'customer-churn-pipeline'
GITHUB_REPO    = os.getenv('GITHUB_REPO',
    'https://github.com/YOUR_USERNAME/customer-churn-mlops.git')

def create_pipeline():
    session = sagemaker.Session()
    role    = sagemaker.get_execution_role()
    env     = {'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY,
               'AWS_SECRET_ACCESS_KEY': AWS_SECRET_KEY,
               'S3_BUCKET': BUCKET, 'AWS_REGION': REGION}

    # Step 1: Preprocess
    processor = SKLearnProcessor(
        framework_version='1.0-1', role=role,
        instance_type='ml.m5.large', instance_count=1, env=env)

    processing_step = ProcessingStep(
        name='PreprocessData', processor=processor,
        code='src/preprocess.py',
        inputs=[sagemaker.processing.ProcessingInput(
            source=f's3://{BUCKET}/data/current/',
            destination='/opt/ml/processing/input')],
        outputs=[sagemaker.processing.ProcessingOutput(
            output_name='processed',
            source='/opt/ml/processing/output',
            destination=f's3://{BUCKET}/data/processed/')])

    # Step 2: Train
    estimator = SKLearn(
        entry_point='src/train.py',
        framework_version='1.0-1', instance_type='ml.m5.large',
        role=role, output_path=f's3://{BUCKET}/models/sagemaker/',
        environment=env)

    training_step = TrainingStep(
        name='TrainModel', estimator=estimator,
        inputs={'train': sagemaker.inputs.TrainingInput(
            s3_data=f's3://{BUCKET}/data/06_encoded_tree.csv',
            content_type='text/csv')})

    pipeline = Pipeline(name=PIPELINE_NAME,
                        steps=[processing_step, training_step])
    pipeline.upsert(role_arn=role)
    print(f"Pipeline created: {PIPELINE_NAME}")
    return pipeline

def trigger_pipeline():
    sm   = boto3.client('sagemaker', region_name=REGION,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY)
    resp = sm.start_pipeline_execution(PipelineName=PIPELINE_NAME,
                                       PipelineExecutionDisplayName='manual-trigger')
    print(f"Pipeline triggered: {resp['PipelineExecutionArn']}")
    return resp['PipelineExecutionArn']

if __name__ == '__main__':
    create_pipeline()
