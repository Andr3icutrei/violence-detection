import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DatasetItem } from './dataset-item';

describe('DatasetItem', () => {
  let component: DatasetItem;
  let fixture: ComponentFixture<DatasetItem>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DatasetItem],
    }).compileComponents();

    fixture = TestBed.createComponent(DatasetItem);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
