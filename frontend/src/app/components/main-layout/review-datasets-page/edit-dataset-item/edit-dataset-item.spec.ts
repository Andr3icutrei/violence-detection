import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EditDatasetItem } from './edit-dataset-item';

describe('EditDatasetItem', () => {
  let component: EditDatasetItem;
  let fixture: ComponentFixture<EditDatasetItem>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EditDatasetItem],
    }).compileComponents();

    fixture = TestBed.createComponent(EditDatasetItem);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
