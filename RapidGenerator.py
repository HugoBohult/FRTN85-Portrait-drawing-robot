
#Plane finder sequence
def plane_finder_sequence(file):
    file.write("\n\tPROC plane_finder()\n")
    file.write("\t\tMoveJ air1,v50,z1,tool1\WObj:=Workobject_1;\n")
    file.write("\t\tSearchL \Stop, DI10_0, searchpoint1, planepoint1, v20, tool1\WObj:=Workobject_1;\n")
    file.write("\t\tMoveJ air2,v50,z1,tool1\WObj:=Workobject_1;\n")
    file.write("\t\tSearchL \Stop, DI10_0, searchpoint2, planepoint2, v20, tool1\WObj:=Workobject_1;\n")
    file.write("\t\tMoveJ air3,v50,z1,tool1\WObj:=Workobject_1;\n")
    file.write("\t\tSearchL \Stop, DI10_0, searchpoint3, planepoint3, v20, tool1\WObj:=Workobject_1;\n")
    file.write("\t\tMoveJ air3,v50,z1,tool1\WObj:=Workobject_1;\n")
    file.write("\t\tWorkobject_1.oframe := DefDframe (planepoint1, planepoint2, planepoint3, searchpoint1, searchpoint2, searchpoint3);\n")
    file.write("\tENDPROC\n\n")

#Define points
def define_Calibpoints(file):
    file.write("\tCONST robtarget air1:=[[500,0,500],[1,0,0,0],[0,0,0,0],[0,0,0,0,0,0]];\n")
    file.write("\tCONST robtarget air2:=[[0,500,500],[1,0,0,0],[0,0,0,0],[0,0,0,0,0,0]];\n")
    file.write("\tCONST robtarget air3:=[[-500,0,500],[1,0,0,0],[0,0,0,0],[0,0,0,0,0,0]];\n")
    file.write("\tCONST robtarget searchpoint1:=[[500,0,100],[1,0,0,0],[0,0,0,0],[0,0,0,0,0,0]];\n")
    file.write("\tCONST robtarget searchpoint2:=[[0,500,100],[1,0,0,0],[0,0,0,0],[0,0,0,0,0,0]];\n")
    file.write("\tCONST robtarget searchpoint3:=[[-500,0,100],[1,0,0,0],[0,0,0,0],[0,0,0,0,0]];\n")
    file.write("\tCONST robtarget planepoint1:=[[500,100,-100],[1,0,0,0],[0,0,0.707107,-.707107],[180.000,-180.000,-90.000]];\n")
    file.write("\tCONST robtarget planepoint2:=[[100,-500,-100],[1.000000e+00  ,  4.329780e-17  ,  7.071068e-01  , -7.071068e-01],[  5.000000e+01  , -1.800000e+02  , -9.000000e+01]];\n")
    file.write("\tCONST robtarget planepoint3:=[[-500,-100,-100],[1.000000e+00  ,  4.329780e-17  ,  7.071068e-01  , -7.071068e-01],[  1.800000e+02  , -1.800000e+02  , -9.000000e+01]];\n\n")

def define_Drawingpoints(file,pointlist):
    for i, point in enumerate(pointlist):
        name = f"Dpoint{i}"
        file.write(stringify_point(name, point[0], point[1]))
        
def stringify_point(name, x, y):
    return f"\tCONST robtarget {name}:=[[{x},{y},0],[1,0,0,0],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n"



#main program
def main():
    file = open("RAPID.txt", "w")
    file.write("MODULE Module1\n")
    file.write("\tCONST robtarget home:= [[0,0,500],[1,0,0,0],[0,0,0,0],[0,0,0,0,0,0]];\n\n")

    #Runs the plane finder sequence
    define_Calibpoints(file)
    define_Drawingpoints(file, [(100,100), (200,200), (300,300)])  # Example drawing points
    plane_finder_sequence(file)

    file.write("\tPROC main()\n")
    file.write("\t\tMoveJ home,v100,z1,tool1\WObj:=Workobject_1;\n")
    file.write("\t\tplane_finder();\n")

    #TO DO: Add drawing sequence here

    file.write("\t\tMoveJ home,v100,z1,tool1\WObj:=Workobject_1;\n")
    file.write("\tENDPROC\n")
    file.write("ENDMODULE\n")

if __name__ == "__main__":
    main()
