import { DatasetModel } from './dataset.model';

export interface Video {
  id: number;
  uid: string;
  name: string;
  path: string;
  is_violent: boolean;
  duration: number;
  frameRate: number;
  dataset: DatasetModel;
}
