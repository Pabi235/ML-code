import time
import logging
import argparse
import datetime
from datetime import datetime
from google.cloud import pubsub
import numpy as np

TOPIC = 'SimulateNormals'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
CUM_MAX_EVENTS = 500

def publish(publisher, topic, events,sleepMeanTime = 5):
    if len(events) > 0:
        logging.info('Publishing {0} events from {1}'.format(len(events),(datetime.utcnow().strftime(TIME_FORMAT))))
        for event_data in events:
            sleep_amount = np.random.randint(sleepMeanTime)
            time.sleep(sleep_amount)
            event_json = {'Variate':event_data,'Timestamp':datetime.utcnow()}
            publisher.publish(topic,event_json)
    
         
def SimulateNormalVariates(topic, mean = 0.0, variance = 1.0,sleepMeanTime = 5,CUM_MAX_EVENTS = 100 ):
    
    counts_eventsPublished = 0

    while counts_eventsPublished <= CUM_MAX_EVENTS:

        Num_events = np.random.randint(50,100,1)
        counts_eventsPublished = counts_eventsPublished + Num_events
        NormalVariates = np.random.normal(loc = mean, scale = variance**0.5, size = Num_events)
        publish(publisher=publisher ,topic=topic,events= NormalVariates,sleepMeanTime = sleepMeanTime)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Send sensor data to Cloud Pub/Sub in small groups, simulating real-time behavior')
    parser.add_argument('--project', help='Example: --project $DEVSHELL_PROJECT_ID', required=True)
    parser.add_argument('--Mean', help='Mean of normal distribution', required=False, type=float,default=0.0)
    parser.add_argument('--Variance', help='Variance of normal distribution', required=False, type=float,default=1.0)
    parser.add_argument('--CumulativeMaxEvents', help='Mean of normal distribution', required=False, type=int,default=100)
    parser.add_argument('--sleepMeanTime', help='Mean of normal distribution', required=False, type=int,default=5)

    args = parser.parse_args()

    # create Pub/Sub notification topic
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    publisher = pubsub.PublisherClient()
    event_type = publisher.topic_path(args.project,TOPIC)
    try:
        publisher.get_topic(event_type)
        logging.info('Reusing pub/sub topic {}'.format(TOPIC))
    except:
        publisher.create_topic(event_type)
        logging.info('Creating pub/sub topic {}'.format(TOPIC))

    # notify about each line in the input file
    logging.info('Sending sensor data from {}'.format(datetime.utcnow().strftime(TIME_FORMAT)))

    SimulateNormalVariates(topic = event_type, mean = args.Mean, variance = args.Variance,sleepMeanTime = args.sleepMeanTime,
                           CUM_MAX_EVENTS = args.CumulativeMaxEvents )



