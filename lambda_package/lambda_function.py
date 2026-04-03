
import json
import boto3
import os
from datetime import datetime

ENDPOINT   = "heart-disease-endpoint"
TABLE_NAME = "health-predictions"
TOPIC_ARN  = os.environ["TOPIC_ARN"]
REGION     = os.environ["AWS_REGION"]

runtime  = boto3.client("sagemaker-runtime", region_name=REGION)
dynamodb = boto3.resource("dynamodb",         region_name=REGION)
sns      = boto3.client("sns",                region_name=REGION)

def lambda_handler(event, context):
    if "body" in event:
        body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
    else:
        body = event

    patient_data = body["patient_data"]
    patient_id   = body.get("patient_id", "unknown")

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="text/csv",
        Body=",".join(map(str, patient_data))
    )
    prob       = float(response["Body"].read().decode("utf-8"))
    prediction = "HIGH_RISK" if prob >= 0.5 else "LOW_RISK"
    timestamp  = datetime.utcnow().isoformat()

    # Store in DynamoDB
    table = dynamodb.Table(TABLE_NAME)
    table.put_item(Item={
        "patient_id":   patient_id,
        "timestamp":    timestamp,
        "risk_score":   str(round(prob, 4)),
        "prediction":   prediction,
        "patient_data": str(patient_data)
    })

    # Send SNS alert if high risk
    if prediction == "HIGH_RISK":
        sns.publish(
            TopicArn=TOPIC_ARN,
            Subject="Health Risk Alert",
            Message=f"HIGH RISK detected!\nPatient ID: {patient_id}\nRisk Score: {prob:.4f}\nTime: {timestamp}"
        )

    result = {
        "patient_id": patient_id,
        "risk_score": round(prob, 4),
        "prediction": prediction,
        "timestamp":  timestamp
    }

    # Return proper API Gateway response
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(result)
    }
