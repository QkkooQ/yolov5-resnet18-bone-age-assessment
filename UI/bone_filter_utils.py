import torch

def filter(box,cls,flag):
    box = box[torch.where(box[:,5] == cls)]
    index = box[:,0].argsort()
    return box[index][flag]

def bone_filter(box):
    if box.shape[0] == 21:
        DistalPhalanx = filter(box,6,[0,2,4])
        MiddlePhalanx = filter(box,5,[0,2])
        MCP = filter(box,4,[0,2,4])
        ProximalPhalanx = filter(box,3,[0,2])
        MCPFirst = filter(box,2,[0])
        Radius = filter(box,1,[0])
        Ulna = filter(box,0,[0])
        return torch.cat([DistalPhalanx,MiddlePhalanx,MCP,ProximalPhalanx,MCPFirst,Radius,Ulna],0)
    else:
        print("关节数出错")