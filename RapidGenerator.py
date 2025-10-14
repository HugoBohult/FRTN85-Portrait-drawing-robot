#Plane finder sequence
def plane_finder_sequence(file):
    file.write("\n\tPROC plane_finder()\n")
    file.write("\t\tMoveJ air1,v50,z1,BHH_pen\WObj:=Workobject_1;\n")
    file.write("\t\tSearchL \Stop, DI10_0, planepoint1, searchpoint1, v20, BHH_pen\WObj:=Workobject_1;\n")
    file.write("\t\tMoveJ air2,v50,z1,BHH_pen\WObj:=Workobject_1;\n")
    file.write("\t\tSearchL \Stop, DI10_0, planepoint2, searchpoint2, v20, BHH_pen\WObj:=Workobject_1;\n")
    file.write("\t\tMoveJ air3,v50,z1,BHH_pen\WObj:=Workobject_1;\n")
    file.write("\t\tSearchL \Stop, DI10_0, planepoint3, searchpoint3, v20, BHH_pen\WObj:=Workobject_1;\n")
    file.write("\t\tMoveJ air3,v50,z1,BHH_pen\WObj:=Workobject_1;\n")
    file.write("\t\tWorkobject_1.oframe := DefDframe (searchpoint1, searchpoint2, searchpoint3,planepoint1, planepoint2, planepoint3);\n")
    file.write("\tENDPROC\n\n")

#Define points
def define_Calibpoints(file):
    file.write("\tCONST robtarget air1:=[[20,20,-160],[1,0,0,0],[0,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n")
    file.write("\tCONST robtarget air2:=[[100,100,-160],[1,0,0,0],[0,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n")
    file.write("\tCONST robtarget air3:=[[160,40,-160],[1,0,0,0],[0,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n")
    file.write("\tCONST robtarget searchpoint1:=[[20,20,0],[1,0,0,0],[0,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n")
    file.write("\tCONST robtarget searchpoint2:=[[100,100,0],[1,0,0,0],[0,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n")
    file.write("\tCONST robtarget searchpoint3:=[[160,40,0],[1,0,0,0],[0,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n")
    file.write("\tVAR robtarget planepoint1;\n")
    file.write("\tVAR robtarget planepoint2;\n")
    file.write("\tVAR robtarget planepoint3;\n\n")

def define_Drawingpoints(file,countours):
    for i, cnt in enumerate(countours):
        for j, point in enumerate(cnt):
            name = f"Dpoint{i}_{j}"
            file.write(stringify_point(name, point[0], point[1]))

def stringify_point(name, x, y):
    return f"\tCONST robtarget {name}:=[[{x},{y},0],[1,0,0,0],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n"

def drawing_sequence(file, countours):
    file.write("\n\tPROC drawing_sequence()\n")
    file.write("\t\tMoveJ Dpoint0,v100,z1,BHH_pen\WObj:=Workobject_1;\n")

    for i, cnt in enumerate(countours):
        file.write(f"\t\tMoveJ Dpoint{i}_{0},v100,z1,BHH_pen\WObj:=Workobject_1;\n")
        for j, point in enumerate(cnt):
            file.write(f"\t\tMoveL Dpoint{i}_{j},v100,z1,BHH_pen\WObj:=Workobject_1;\n")
        file.write(f"\t\tMoveJ Dpoint{i}_{len(cnt)-1},v100,z1,BHH_pen\WObj:=Workobject_1;\n")

    file.write("\tENDPROC\n\n")



#main program
def generator(countours):
    file = open("RAPID.txt", "w")
    file.write("MODULE Module1\n")
    file.write("\tCONST robtarget home:= [[280.027029643,125.4,-515.328184811],[0,0.608761442,0,0.79335333],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n\n")

    #Runs the plane finder sequence
    define_Calibpoints(file)
    define_Drawingpoints(file, countours)
    plane_finder_sequence(file)
    drawing_sequence(file, countours)  # Example drawing points

    file.write("\tPROC main()\n")
    file.write("\t\tMoveJ home,v100,z1,BHH_pen\WObj:=Workobject_1;\n")
    file.write("\t\tplane_finder;\n")
    file.write("\t\tdrawing_sequence;\n")

    file.write("\t\tMoveJ home,v100,z1,BHH_pen\WObj:=Workobject_1;\n")
    file.write("\tENDPROC\n")
    file.write("ENDMODULE\n")

if __name__ == "__main__":
    test_points = [(20, 20), (50, 50), (30, 50)]
    main(test_points)
