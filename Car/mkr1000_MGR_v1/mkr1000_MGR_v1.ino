
/*
 MGR control loop based on MQTT communication

  - connects to an MQTT server, providing username
    and password
  - subscribes to command topic
  - executes command by updating output to motors driver
*/

#include <SPI.h>
#include <WiFi101.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const byte M1PWM = 2;
const byte M1PH = 3;
const byte M2PWM = 4;
const byte M2PH = 5;
const byte MODE_PIN = 7;

int throt_A = 0;
int throt_B = 0;
bool dir_A = LOW;
bool dir_B = LOW;

// WIFI:
const char ssid[] = "network_name";
const char pass[] = "password";


IPAddress server(111,111,2,111); //station IP (localhost)
const char topic_c[] = "mkr/command";
const char topic_s[] = "mkr/setup";
const char ID[] = "1"; // unique ID in the group


const int ledPin =  LED_BUILTIN;// the number of the LED pin



//--------------------------------------- callback -------------------------------

void callback(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<256> doc;
  deserializeJson(doc, payload, length);
  throt_A = doc[ID]["L"];
  throt_B = doc[ID]["R"];
  execute();  
}


WiFiClient net;
PubSubClient client(server, 1883, callback, net);



//--------------------------------------- functions -----------------------------

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (client.connect(ID)) {
      Serial.println("connected");
      // Once connected, publish an announcement...
           
      char  temp[3]; //temporal data
      sprintf(temp, ID);
      
      client.publish(topic_s, temp);
      // ... and resubscribe
      client.subscribe(topic_c);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}



void execute() //received command execution 
{
  if (throt_A < 0) {
    dir_A = LOW;
  }
  else {
    dir_A = HIGH;
  }
  if (throt_B < 0) {
    dir_B = LOW;
  }
  else {
    dir_B = HIGH;
  }
  digitalWrite(M1PH, dir_A);
  digitalWrite(M2PH, dir_B);
  analogWrite(M1PWM, abs(throt_A));
  analogWrite(M2PWM, abs(throt_B));
}
  

//--------------------------------------- setup ---------------------------------


void setup()
{
  
  Serial.begin(115200);
  WiFi.begin(ssid, pass);
  // Note - the default maximum packet size is 128 bytes. If the
  // combined length of clientId, username and password exceed this use the
  // following to increase the buffer size:
  client.setBufferSize(255);
  pinMode(ledPin, OUTPUT);
  pinMode(MODE_PIN, OUTPUT);
  digitalWrite(MODE_PIN, HIGH);
  analogWrite(M1PWM, 0);
  analogWrite(M2PWM, 0);

}


//--------------------------------------- main loop -----------------------------

void loop()
{
  
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

}
