import pandas as pd

def html_from_template(template_file, out_file, **kwargs):
    # print('load template file', template_file)
    with open(template_file, "r") as fin:
        lines = fin.readlines()
    template_string = "".join(lines)
    outfile_string = template_string.format(**kwargs)
    # print(f'Export html to {out_file}')
    with open(out_file, "w") as fout:
        fout.write(outfile_string)


def logger_table_html(result, out_file, template_file="", suffix=""):
    """Result table html
              obj_index(link)  | Loss
         0       shape_id       0.01
         1         ...           ...

       Notes:
           Pandas library is used to convert table data to html string.
    """
    obj_data_format = """<a href="./{obj}/{obj}{suffix}.html" title="{obj}"> {obj} </a>"""

    table = []
    for item in result:
        obj_name, loss = item
        table.append([obj_data_format.format(obj=obj_name, suffix=suffix),
                      loss])

    df = pd.DataFrame(table, columns=['Obj', 'Loss-CD'])
    tmp = df.to_html(escape=False, classes="table sortable is-striped is-hoverable", border=0)

    html_from_template(template_file,
                       out_file=out_file,
                       table_string=tmp)


if __name__ == "__main__":
    template_fn = '/Users/sfhan/Documents/project/voxelPoint/tmp/viewer/pc_vis_template.html'
    out_fn = '/Users/sfhan/Documents/project/voxelPoint/tmp/viewer/pc_vis_test.html'
    html_from_template(template_fn, out_fn, shape_id=1,
                       input_fn='./data/input.pcd', pred_fn='./data/pred.pcd',
                       gt_fn='./data/gt.pcd')
