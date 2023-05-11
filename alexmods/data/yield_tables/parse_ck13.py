from alexmods.smhutils import element_to_species
fp = open("YIELD_CK13.DAT", "r")
newfp = open("nkt13.dat","w")
# Output columns:
# Zmet, M, E, Mrem, Z, A, yield
fmt = "{:.4f}, {:.1f}, {:.1f}, {:.3f}, {}, {}, {:.6e}\n"

line = fp.readline()
while True:
    if line=="": break
    if line.startswith("Z="):
        Zmet = round(float(line.split()[1]),4)
        Mline = fp.readline().split()[1:]
        Eline = fp.readline().split()[1:]
        Mremline = fp.readline().split()[1:]
        assert len(Mline)==len(Eline)
        assert len(Mremline)==len(Mline)
        while True:
            line = fp.readline()
            if line.startswith("Z="): break
            if line=="": break
            line = line.split()
            elem = line[0]
            Z = int(element_to_species(elem))
            A = int(line[1])
            for i, (M, E, Mrem) in enumerate(zip(Mline, Eline, Mremline)):
                newfp.write(fmt.format(Zmet, float(M), float(E), float(Mrem), Z, A, float(line[2+i])))
        
