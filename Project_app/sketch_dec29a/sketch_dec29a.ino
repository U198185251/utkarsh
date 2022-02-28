int b = A0;
int g = A1;
int r = A2;

          // the PWM pin the LED is attached to
int brightness = 0;    // how bright the LED is
int fadeAmount = 5;    // how many points to fade the LED by

void setup() { 
  Serial.begin(38400);
  Serial.println("Arduino is ready");//initialize serial COM at 9600 baudrate
  pinMode(b, OUTPUT);                    //declare the LED pin (13) as output
  pinMode(g, OUTPUT);
  pinMode(r, OUTPUT);
  analogWrite(b, 0);                     //Turn OFF the Led in the beginning
  analogWrite(g, 0);
  analogWrite(r, 0);
}
void loop() {
while (Serial.available())   //whatever the data that is coming in serially and assigning the value to the variable “data”
{ 
char data = Serial.read();
Serial.println(data);
if (data=='1')
{
  analogWrite(b,250);
  analogWrite(g,0);
  analogWrite(r,0);
  
} // end switch
else if (data=='2'){
  analogWrite(b,50);
  analogWrite(g,200);
  analogWrite(r,50);
  
    }
else if (data=='3'){
  analogWrite(b,0);
  analogWrite(g,0);
  analogWrite(r,200);
  
    }
else if (data=='4'){
  analogWrite(b,50);
  analogWrite(g,200);
  analogWrite(r,200);
  
    }
else if (data=='5'){
  analogWrite(b,250);
  analogWrite(g,250);
  analogWrite(r,250);
  
    }

else if (data=='6'){
  analogWrite(b,250);
  analogWrite(g,100);
  analogWrite(r,100);
  
    }
else if (data=='7'){
  analogWrite(b,250);
  analogWrite(g,0);
  analogWrite(r,200);
  
    }
  
  } //end while
} // end main loop
